# littera: working rules

- Plan, decisions, architecture: `docs/PLAN.md`. Round documents: `docs/r1/`, `docs/r2/`. Shared output type: `src/littera/types.py`
  (`Placement`, schema `littera/placement/1`); change it only through the orchestrator.
- Python 3.12 with uv: `uv run python ...`, `uv run pytest`. Local torch is the CPU build. Do not edit `pyproject.toml`
  or `uv.lock` during a round; Modal images install whatever they need.
- Local machine: light work only (code, unit tests, small samples). Data processing, training and bulk inference run
  on Modal. Never open plot windows: `matplotlib.use("Agg")`.
- Modal: `PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/<file>.py::<entry>`; on Windows pass paths as `C:/...`,
  never `/c/...`. Apps are named `littera-<area>`, the volume is `littera` mounted at `/vol`. At most 10 containers
  at once, CPU unless a GPU is needed, `--detach` for long jobs. Read-only use of other projects' volumes:
  `kurrent-raw` (line pool), `humboldt-htr` (aligner-k2 weights). Report spend from `modal billing report --for today`.
- Training data must be openly licensed (MIT, CC BY, CC BY-SA, CC0, public domain mark). Everything else (NC, ND,
  internal archival material, registration-only sets) may only be used for evaluation.
- Code: clear names and plain logic, almost no comments or docstrings, ruff line length 140.
- Git: agents do not commit. The orchestrator commits as the owner (git identity Maelkolb), without any co-author trailer.
