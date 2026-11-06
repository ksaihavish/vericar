"""Run the whole offline pipeline (or a subset) in order, logging each step.

    python scripts/run_all.py                              # everything
    python scripts/run_all.py --steps fit_hybrid,optimize  # a subset, in pipeline order

Each step's output goes to VERICAR_WORK/runs/<step>.log; progress and exit codes
to VERICAR_WORK/runs/pipeline.log. Stops at the first failing step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vericar.config import RUNS_DIR  # noqa: E402

ORDER = ["prepare_data", "train_detector", "train_classifier", "train_classical", "fit_hybrid", "optimize",
         "evaluate_final", "make_assets"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(ORDER))
    ap.add_argument("--classifier-args", default="")
    a = ap.parse_args()
    wanted = [s for s in ORDER if s in a.steps.split(",")]
    plog = RUNS_DIR / "pipeline.log"
    for step in wanted:
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / f"{step}.py")]
        if step == "train_classifier" and a.classifier_args:
            cmd += a.classifier_args.split()
        with open(plog, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%H:%M:%S} START {step}\n")
        with open(RUNS_DIR / f"{step}.log", "w", encoding="utf-8") as out:
            rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
        with open(plog, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%H:%M:%S} END {step} rc={rc}\n")
        if rc != 0:
            return rc
    with open(plog, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%H:%M:%S} ALL DONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
