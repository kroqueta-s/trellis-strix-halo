# SPDX-License-Identifier: MIT
"""Write a pipeline description that names local files only.

`Pipeline.from_pretrained` takes the config file's name as an argument, so
**the downloaded `pipeline.json` is never edited** and this one sits beside it.
Three things change:

- the texture models are kept **only when their checkpoints are there**, so a
  geometry-only install is described as geometry-only rather than failing to
  load a file it never downloaded;
- the image conditioner and the background remover point at directories on this
  machine rather than at hub repositories, **one of which is gated** - and a
  path an earlier run already wrote is kept if it still exists, because an
  operator may have pointed one at a copy shared with another runner;
- the sparse-structure decoder points at the copy in the weights directory.

    python tools\\write_local_pipeline.py C:\\dev\\models\\trellis2
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any


def write_local_pipeline(weights: Path) -> Path:
    """Write `pipeline.local.json` beside the downloaded description.

    Args:
        weights: The directory holding `pipeline.json` and `ckpts/`.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the downloaded description is not there.
    """
    source = weights / "pipeline.json"
    if not source.is_file():
        raise FileNotFoundError(f"no pipeline.json in {weights} (were the weights downloaded?)")

    config = json.loads(source.read_text(encoding="utf-8"))
    args = config["args"]

    # **A missing checkpoint is dropped, not left to fail at load time.**
    # `from_pretrained` walks this table and loads every entry, so naming a file
    # that was never downloaded turns into an exception inside the pipeline -
    # and that one is swallowed (see `docs/trellis2.md`).
    dropped = [
        name
        for name, ckpt in args["models"].items()
        if name.startswith("tex_") and not (weights / f"{ckpt}.safetensors").is_file()
    ]
    for name in dropped:
        del args["models"][name]
    args["models"]["sparse_structure_decoder"] = "ckpts/ss_dec_conv3d_16l8_fp16"

    target = weights / "pipeline.local.json"
    # **What is already there wins, as long as it is still there.** These two
    # are the only settings an operator has a reason to move by hand - BiRefNet
    # in particular is often shared with another runner rather than downloaded
    # twice - and regenerating this file for an unrelated reason should not
    # quietly point them somewhere empty.
    previous: dict[str, Any] = {}
    if target.is_file():
        with contextlib.suppress(ValueError, KeyError, OSError):
            previous = json.loads(target.read_text(encoding="utf-8"))["args"]

    for section, local in (
        ("image_cond_model", weights / "dinov3-vitl16-pretrain-lvd1689m"),
        ("rembg_model", weights / "BiRefNet"),
    ):
        entry = args[section]
        key = "model_name" if "model_name" in entry["args"] else next(iter(entry["args"]))
        kept = previous.get(section, {}).get("args", {}).get(key)
        entry["args"][key] = kept if kept and Path(kept).is_dir() else str(local)
        print(f"{section}: {entry['args'][key]}")

    target.write_text(json.dumps(config, indent=4), encoding="utf-8")
    print(f"texture models without a checkpoint, dropped: {dropped}")
    print(f"models: {sorted(args['models'])}")
    print(f"wrote {target}")
    return target


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    write_local_pipeline(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
