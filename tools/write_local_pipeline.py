# SPDX-License-Identifier: MIT
"""Write a pipeline description that names local files only.

`Pipeline.from_pretrained` takes the config file's name as an argument, so
**the downloaded `pipeline.json` is never edited** and this one sits beside it.
Three things change:

- the texture models are dropped, because this runner returns geometry;
- the image conditioner and the background remover point at directories on this
  machine rather than at hub repositories, **one of which is gated**;
- the sparse-structure decoder points at the copy in the weights directory.

    python tools\\write_local_pipeline.py C:\\dev\\models\\trellis2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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

    dropped = [name for name in args["models"] if name.startswith("tex_")]
    for name in dropped:
        del args["models"][name]
    args["models"]["sparse_structure_decoder"] = "ckpts/ss_dec_conv3d_16l8_fp16"

    for section, local in (
        ("image_cond_model", weights / "dinov3-vitl16-pretrain-lvd1689m"),
        ("rembg_model", weights / "BiRefNet"),
    ):
        entry = args[section]
        key = "model_name" if "model_name" in entry["args"] else next(iter(entry["args"]))
        entry["args"][key] = str(local)

    target = weights / "pipeline.local.json"
    target.write_text(json.dumps(config, indent=4), encoding="utf-8")
    print(f"dropped the texture models: {dropped}")
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
