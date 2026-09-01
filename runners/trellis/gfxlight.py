# SPDX-License-Identifier: MIT
"""「3D の常夜灯」：非表示ウィンドウで軽い OpenGL 描画を回し、GPU を高い電力ステートに保つ。

**なぜ要るか**（2026-09-01 実測・docs/02_port_report.md）：Windows の AMD ドライバは
compute キューだけの負荷では電力ステート（DPM）を上げない。GPU 使用率 99% でも
GFXCLK 600 MHz に張り付き、GEMM は 4.8 TFLOPS しか出ない。3D 描画が 1 つ動いていれば
2.35 GHz へ上がり、同じ GEMM が **20.9 TFLOPS（4.3 倍）** になる。非表示ウィンドウでも効く。
MyASUS 等の常駐 UI がたまたま描画していると速い——という生成時間のばらつき（3.9 倍）の
原因もこれで、常夜灯はそれを決定的にする。

**設計**：ctypes だけで書く（追加パッケージなし・自前バイナリなし＝Smart App Control に
掛からない）。子プロセスとして走らせ、親が死ねば自分で消える。点かなくても生成は
従来どおり動く（効果が metrics.gfx_keepalive に載るだけ）。

単体での確認（どのランナーの python でもよい。torch を使わない）::

    python -m runners.<runner>.gfxlight --seconds 10

このファイルは 3 ランナーで同一内容（各ランナーは独立リポジトリへ出すため共有しない）。
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


# --- 親（ランナー）側 ---------------------------------------------------------
class GfxLight:
    """常夜灯の子プロセスを点けたり消したりする。**点かなくても例外は投げない。**"""

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        """子プロセスで常夜灯を点ける。失敗しても生成は続行する。"""
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
            print(f"[gfxlight] 点かない（生成は続行する）: {exc}", file=sys.stderr)
            self._proc = None

    def is_lit(self) -> bool:
        """今も点いているか。metrics に載せて、効いていたかを後から判定できるようにする。"""
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        """消す。子は親の死でも自分で消えるが、正常系では明示的に消す。"""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


# --- 子（描画ループ）側 -------------------------------------------------------
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

    # **表示しない。** 非表示でも DPM は上がる（実測：GEMM 4.8 → 20.9 TFLOPS）。
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

    # ソフトウェア描画に落ちていたら GPU に効かないので、点いていない扱いで消える。
    renderer = opengl32.glGetString(GL_RENDERER) or b""
    if b"GDI Generic" in renderer:
        return 3

    # 親が死んだら消える。フレーム間隔の待ちを親プロセスの監視で兼ねる。
    #
    # **`glFinish` はしない・1 フレームの描画量をわざと多めにする。**
    # 長い compute カーネル（Hunyuan3D の DiT 等）が GPU を独占すると 3D キューが
    # 飢餓になり、同期待ちのある極小の描画では「3D の需要」がドライバから見えなく
    # なってクロックが 600 MHz へ落ちた（2026-09-02 実測：gen 179 s・keepalive 生存中）。
    # 同期せずにフレームを先行投入し、キューに常に仕事が積まれた状態を保つ。
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
                break  # 親が死んだ
        else:
            time.sleep(wait_ms / 1000.0)

    opengl32.wglMakeCurrent(None, None)
    opengl32.wglDeleteContext(ctx)
    user32.ReleaseDC(hwnd, dc)
    user32.DestroyWindow(hwnd)
    return 0


def main() -> int:
    """子プロセスとして描画ループを回す（単体実行は --seconds で時間を切る）。"""
    parser = argparse.ArgumentParser(description="3D の常夜灯（詳細はモジュール docstring）")
    parser.add_argument("--parent-pid", type=int, default=0, help="このプロセスが死んだら消える")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 なら親が死ぬまで")
    args = parser.parse_args()
    return _render_loop(args.parent_pid, args.fps, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
