# SPDX-License-Identifier: MIT
"""Clock keepalive: a hidden OpenGL render loop that holds the GPU in a high power state.

**Why this is needed** (measured 2026-09-01): the AMD Windows driver does not
raise the GPU power state (DPM) for compute-only work. At 99 % GPU utilisation
the clock sits at 600 MHz and GEMM reaches only 4.8 TFLOPS. With any 3D
rendering alive alongside it, the clock rises to 2.35 GHz and the same GEMM
reaches **20.9 TFLOPS (4.3x)**. A hidden window is enough. This also explains
the 3.9x spread in generation time: runs were fast only when some resident UI
happened to be drawing. The keepalive makes that the normal case.

**Design**: written with ctypes only (no extra packages, no self-built binaries,
so Smart App Control has nothing to block). It runs as a child process and exits
on its own when the parent dies. If it fails to start, generation proceeds
exactly as before; the difference only shows up in `metrics.gfx_keepalive`.

Check it on its own (any runner's python works; torch is not used)::

    python -m runners.<runner>.gfxlight --seconds 10

This file is identical in all three runners, which do not share a module because
each ships as its own repository.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

WS_POPUP = 0x80000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

PFD_DOUBLEBUFFER = 0x00000001
PFD_DRAW_TO_WINDOW = 0x00000004
PFD_SUPPORT_OPENGL = 0x00000020
PFD_TYPE_RGBA = 0

GL_COLOR_BUFFER_BIT = 0x00004000
GL_TRIANGLES = 0x0004
GL_RENDERER = 7937

SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102

WNDPROC = ctypes.WINFUNCTYPE(
    ctypes.c_longlong, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM
)


class _WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", ctypes.c_void_p),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class _PIXELFORMATDESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ("nSize", wintypes.WORD),
        ("nVersion", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("iPixelType", ctypes.c_ubyte),
        ("cColorBits", ctypes.c_ubyte),
        ("cRedBits", ctypes.c_ubyte),
        ("cRedShift", ctypes.c_ubyte),
        ("cGreenBits", ctypes.c_ubyte),
        ("cGreenShift", ctypes.c_ubyte),
        ("cBlueBits", ctypes.c_ubyte),
        ("cBlueShift", ctypes.c_ubyte),
        ("cAlphaBits", ctypes.c_ubyte),
        ("cAlphaShift", ctypes.c_ubyte),
        ("cAccumBits", ctypes.c_ubyte),
        ("cAccumRedBits", ctypes.c_ubyte),
        ("cAccumGreenBits", ctypes.c_ubyte),
        ("cAccumBlueBits", ctypes.c_ubyte),
        ("cAccumAlphaBits", ctypes.c_ubyte),
        ("cDepthBits", ctypes.c_ubyte),
        ("cStencilBits", ctypes.c_ubyte),
        ("cAuxBuffers", ctypes.c_ubyte),
        ("iLayerType", ctypes.c_ubyte),
        ("bReserved", ctypes.c_ubyte),
        ("dwLayerMask", wintypes.DWORD),
        ("dwVisibleMask", wintypes.DWORD),
        ("dwDamageMask", wintypes.DWORD),
    ]


# --- Parent (runner) side -----------------------------------------------------
class GfxLight:
    """Turns the keepalive child process on and off. **Never raises when it fails.**"""

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """Start the keepalive child. Generation continues even if this fails."""
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--parent-pid",
                    str(os.getpid()),
                    "--fps",
                    str(self._fps),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            print(f"[gfxlight] not started (generation continues): {exc}", file=sys.stderr)
            self._proc = None

    def is_lit(self) -> bool:
        """Whether it is still alive. Recorded in metrics so a run can be judged later."""
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        """Stop it. The child also exits when the parent dies; this is the clean path."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


# --- Child (render loop) side -------------------------------------------------
def _render_loop(parent_pid: int, fps: float, seconds: float) -> int:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    opengl32 = ctypes.windll.opengl32
    kernel32 = ctypes.windll.kernel32

    user32.DefWindowProcW.restype = ctypes.c_longlong
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        ctypes.c_uint,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.GetDC.restype = ctypes.c_void_p
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
    gdi32.ChoosePixelFormat.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.SetPixelFormat.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    gdi32.SwapBuffers.argtypes = [ctypes.c_void_p]
    opengl32.wglCreateContext.restype = ctypes.c_void_p
    opengl32.wglCreateContext.argtypes = [ctypes.c_void_p]
    opengl32.wglMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    opengl32.wglDeleteContext.argtypes = [ctypes.c_void_p]
    opengl32.glGetString.restype = ctypes.c_char_p
    opengl32.glGetString.argtypes = [ctypes.c_uint]
    opengl32.glClearColor.argtypes = [ctypes.c_float] * 4
    opengl32.glVertex2f.argtypes = [ctypes.c_float, ctypes.c_float]

    proc = WNDPROC(lambda hw, msg, wp, lp: user32.DefWindowProcW(hw, msg, wp, lp))
    hinst = kernel32.GetModuleHandleW(None)
    cls = _WNDCLASSW()
    cls.lpfnWndProc = proc
    cls.hInstance = hinst
    cls.lpszClassName = "hearth_gfxlight"
    if not user32.RegisterClassW(ctypes.byref(cls)):
        return 1

    # **Never shown.** The power state rises even for a hidden window
    # (measured: GEMM 4.8 -> 20.9 TFLOPS).
    hwnd = user32.CreateWindowExW(
        WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        "hearth_gfxlight",
        "hearth gfx light",
        WS_POPUP,
        0,
        0,
        256,
        256,
        None,
        None,
        hinst,
        None,
    )
    if not hwnd:
        return 1

    dc = user32.GetDC(hwnd)
    pfd = _PIXELFORMATDESCRIPTOR()
    pfd.nSize = ctypes.sizeof(_PIXELFORMATDESCRIPTOR)
    pfd.nVersion = 1
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
    pfd.iPixelType = PFD_TYPE_RGBA
    pfd.cColorBits = 32
    fmt = gdi32.ChoosePixelFormat(dc, ctypes.byref(pfd))
    if not fmt or not gdi32.SetPixelFormat(dc, fmt, ctypes.byref(pfd)):
        return 1
    ctx = opengl32.wglCreateContext(dc)
    if not ctx or not opengl32.wglMakeCurrent(dc, ctx):
        return 1

    # Software rendering cannot affect the GPU, so exit and report as not lit.
    renderer = opengl32.glGetString(GL_RENDERER) or b""
    if b"GDI Generic" in renderer:
        return 3

    # Exit when the parent dies; waiting on the parent handle doubles as the
    # frame interval.
    #
    # **Do not call `glFinish`, and keep each frame's workload generous.**
    # When one long compute kernel monopolises the GPU (the Hunyuan3D DiT does),
    # the 3D queue starves. With a tiny synchronised draw the driver then sees
    # no 3D demand at all and drops the clock back to 600 MHz (measured
    # 2026-09-02: 179 s generation with the keepalive still alive). Submitting
    # frames without synchronising keeps 3D work queued at all times.
    parent = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid) if parent_pid else None
    wait_ms = max(1, int(1000.0 / fps))
    deadline = time.time() + seconds if seconds > 0 else None
    frame = 0
    while deadline is None or time.time() < deadline:
        opengl32.glClearColor(0.0, 0.0, (frame % 100) / 100.0, 1.0)
        opengl32.glClear(GL_COLOR_BUFFER_BIT)
        opengl32.glBegin(GL_TRIANGLES)
        for i in range(150):
            x = -1.0 + (i % 25) * 0.08
            y = -1.0 + (i // 25) * 0.32
            opengl32.glVertex2f(x, y)
            opengl32.glVertex2f(x + 0.4, y)
            opengl32.glVertex2f(x, y + 0.4)
        opengl32.glEnd()
        gdi32.SwapBuffers(dc)
        frame += 1
        if parent:
            if kernel32.WaitForSingleObject(parent, wait_ms) != WAIT_TIMEOUT:
                break  # the parent died
        else:
            time.sleep(wait_ms / 1000.0)

    opengl32.wglMakeCurrent(None, None)
    opengl32.wglDeleteContext(ctx)
    user32.ReleaseDC(hwnd, dc)
    user32.DestroyWindow(hwnd)
    return 0


def main() -> int:
    """Run the render loop as a child process (`--seconds` bounds a standalone run)."""
    parser = argparse.ArgumentParser(description="Clock keepalive (see the module docstring)")
    parser.add_argument("--parent-pid", type=int, default=0, help="exit when this process dies")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means until the parent dies")
    args = parser.parse_args()
    return _render_loop(args.parent_pid, args.fps, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
