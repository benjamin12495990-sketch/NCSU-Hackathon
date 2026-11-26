# NCSU-Hackathon: start here (Claude Code sessions)

@docs/AGENTS.md

- The file above is loaded with this one: who owns which folder, the ground rules, and the five questions to answer before
  saying a result is done.
- Before proposing a change of method or evaluation, read `docs/DECISIONS.md` (decisions already made and mistakes already
  caught). Add a row when you make such a change.
- Before training, tuning or scoring a model, read `docs/ISOLATED_MODEL_TRAINING_PROTOCOL.md` (the checklist in its section
  8), and `lab/README.md` or `gpu/README.md` for the folder you work in. `docs/AGENTS.md` has no ownership row for `lab/`,
  `gpu/` or `pipeline/`; those READMEs hold their rules.
- Do not open `lab/vault/` or these five files yourself (no Read, `cat`, `grep` or `head`), and do not write `lab/` or `gpu/`
  code that reads them: `public/model/nn_data.json`, `public/model/known.json`, `public/model/test_points.json`,
  `public/model/parity.json`, `train/data/concrete.csv`. They hold the concrete rows older than 28 days, which were sealed
  and then opened once on 2026-11-21, so they cannot be a fresh test again. `lab/guard.mjs` (inside lab scripts) and
  `guard_path` in `gpu/data.py` (inside the GPU loaders) refuse them, but nothing stops a direct read, so keeping this rule
  is up to you. Files with the same name in subfolders (`public/model/materials/nn_data.json` and so on) are not blocked.
  The live page's code (`train/train.py`, `scripts/parity.mjs`, `src/`) reads them by design; you may run it, but do not
  print or copy their rows.
- Do not delete or overwrite experiment outputs. The lab archive and vault (`lab/archive*/`, `lab/vault/`), `lab/results/`,
  registered models and finished run folders under `gpu/runs/` are write-once: a changed run gets a new folder, and a
  superseded report or registration is renamed, not replaced (`REPORT.<time>.old.md`, `preregistration_v1.json`). This does
  not cover everything: `train/train.py` overwrites `public/model/*.json` in place, checkpoints, caches and status files
  are replaced, `gpu/runs/` is not in git, and code and documents are edited normally (git keeps the history). Ask before
  deleting any file.
