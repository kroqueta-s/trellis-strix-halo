# CLAUDE.md — working on trellis-strix-halo

**TRELLIS image-to-mesh on AMD Strix Halo (gfx1151), Windows, ROCm.**

Read this first, then [`README.md`](README.md), which carries the measurements
and the limits.

**This repository is a runner.** It implements
[hearth's runner contract](https://github.com/kroqueta-s/hearth/blob/main/docs/runner_contract.md):
one JSON object per line over stdin and stdout, `capabilities` answered as data,
counted progress, and an `unload` that gives the VRAM back. **That document is
the specification** — where this repository and the contract disagree, the
contract wins.

**If `docs/local/` exists, read `docs/local/00_operator_notes.md` too.** It holds
this machine's setup, the environment-specific traps, and how the operator wants
to be told things. It is deliberately not tracked (see below).

---

## Language: **what ships is English, what explains is not**

This repository is public. **Assume everything tracked by git will be read by a
stranger.**

| Where | Language | Why |
|---|---|---|
| **Everything tracked by git** — code, comments, docstrings, **every string anyone reads at runtime**, `README.md`, commit messages, `.env.example` | **English** | It is published. A reader who does not share the author's language should not be shut out of the part they have to use. |
| **`docs/local/`** — design notes, verification reports, the record of what was tried and rejected | The author's language | It is **not tracked**, and never will be. It carries the operator's environment, dead ends, and reasoning that would only mislead a stranger reading it as documentation. |

**Text shown to a person is not an exception to this.** Progress messages, error
messages and log lines are English, the same as the code around them. **Assume
whoever runs this reads English**: an error they cannot read is worse than no
error, and the person most likely to be reading one is a stranger who just
installed it.

- **Do not translate the internal documents.** They are not drafts of the public
  ones; they are a different kind of document with a different reader.
- **Do not put internal reasoning into a tracked file** to keep it together with
  the code. If it explains *why this machine*, it belongs in `docs/local/`.

## Architecture: **break these and the design stops working**

1. **Upstream code is never modified.** It is cloned and run as it is, because a
   re-clone or an update silently discards a local edit. **Everything this
   hardware needs is injected at launch time**, from `runners/trellis/shims.py`
   and the rest of the runner — and that is why this repository survives an
   upstream release.
2. **This runner never imports hearth.** It has to stay shippable on its own, and
   it runs standalone through `tools/run_single.py`. The contract is a document,
   not a dependency.
3. **Nothing but the protocol writes to stdout** (contract §1). Model code prints
   as a matter of course, and replacing `sys.stdout` is not enough: a compiled
   extension writes to file descriptor 1 directly. The guard goes in before
   anything else is imported.
4. **`capabilities` answers without loading the model** (contract §2), and it is
   **data**: a caller reads the table rather than checking which model this is.
5. **`unload` gives the VRAM back and reports what is still held.** There is one
   GPU, and a switch that quietly kept it makes everything after it slower.
6. **Report what you counted, never an estimate** (contract §8). On this hardware
   the first run of a loop can be an order of magnitude slower than the ones
   after it, so a prediction built from a stored constant is worst exactly when
   it is most wanted. **No ETA, and no percentage without a real total.**
7. **No path, no port and no tuning constant is written in code.** `.env` is the
   authority, and `.env.example` keeps the same set of keys.
8. **A replacement for an upstream package is checked against upstream's own
   semantics**, not merely made plausible. Submanifold convolution is exactly a
   dense `F.conv3d` restricted to active voxels, so `tests/test_shims.py` checks
   it against a dense reference. **Keep that true of anything new.**

## Style

- Python: `ruff` and `black`, line length 100, **type hints everywhere**.
- **Comments say why, not what.** The code already says what.
- **A default that came from a measurement should say so**, and one that did not
  should say that too.
- Tests are hand-written scripts under `tests/`. **Do not add `pytest`**: this
  environment pins its dependencies exactly.
- **Run the tests after changing anything they cover.**

## Do not

- **Modify upstream's source.** See rule 1; it has never needed an exception.
- Import hearth, or anything from another runner's repository.
- Write anything but protocol to stdout.
- Add a dependency that exists only for CUDA. **The point of this repository is
  that none is needed.**
- Publish, or track in git, anything from `docs/local/`.
