"""BIDS derivative input interface for the Buckner-compatible workflow."""

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys

from pymshbm.io.bids import discover_surface_runs


def main(argv: list[str] | None = None) -> None:
    """Run ``pymshbm bids``; the module also accepts arguments without ``bids``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bids":
        argv.pop(0)
    parser = argparse.ArgumentParser(
        prog="pymshbm bids",
        description="Fit paired fsaverage6 fMRIPrep BOLD runs using the Buckner workflow.",
        epilog="Each physical run is one model session. No additional denoising or resampling is applied.",
    )
    parser.add_argument("input", type=Path, help="fMRIPrep derivative directory or BIDS root.")
    parser.add_argument("output", type=Path, help="New output derivative directory.")
    parser.add_argument("--assets-dir", type=Path, help="Pinned reference assets (required unless --dry-run).")
    parser.add_argument("--participant-label", nargs="+", action="extend", help="Participant labels, with or without sub-.")
    parser.add_argument("--session-label", nargs="+", action="extend", help="Session labels, with or without ses-.")
    parser.add_argument("--run-label", nargs="+", action="extend", help="Run labels, with or without run-.")
    parser.add_argument("--task", default="rest", help="BIDS task label (default: rest).")
    parser.add_argument("--dry-run", action="store_true", help="Print a JSON input manifest without reading images or writing output.")
    parser.add_argument("--max-iter", type=int, default=5, help="Maximum outer model iterations (default: 5).")
    parser.add_argument("--allow-zero-cortex", action="store_true",
                        help="Allow entirely zero nonseed cortical time courses; record coverage and preserve zero profiles.")
    parser.add_argument("--verbose", action="store_true", help="Log optimizer progress in detail.")
    args = parser.parse_args(argv)
    if args.max_iter < 1:
        parser.error("--max-iter must be a positive integer")
    if not args.dry_run and args.assets_dir is None:
        parser.error("--assets-dir is required for fitting; use --dry-run to inspect inputs")
    try:
        runs = discover_surface_runs(args.input, participant_labels=args.participant_label,
                                     session_labels=args.session_label, task=args.task,
                                     run_labels=args.run_label)
        if args.dry_run:
            sessions = defaultdict(int)
            records = []
            for run in runs:
                sessions[run.subject] += 1
                records.append({"subject": run.subject, "session": run.session,
                                "task": run.task, "run": run.run,
                                "model_session": sessions[run.subject],
                                "lh": str(run.lh), "rh": str(run.rh),
                                "entities": dict(run.entities)})
            manifest = {"input_dir": str(args.input.expanduser().resolve()),
                        "output_dir": str(args.output.expanduser().resolve()),
                        "assets_dir": str(args.assets_dir.expanduser().resolve()) if args.assets_dir else None,
                        "max_iter": args.max_iter,
                        "allow_zero_cortex": args.allow_zero_cortex,
                        "preprocessing": {"additional_denoising": False, "resampling": False},
                        "runs": records}
            print(json.dumps(manifest, indent=2))
            return
        from pymshbm.pipeline.buckner import run_buckner_workflow

        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(levelname)s: %(message)s")
        result = run_buckner_workflow(runs, args.output, args.assets_dir, max_iter=args.max_iter,
                                      allow_zero_cortex=args.allow_zero_cortex)
        print(f"Workflow complete. Output: {result}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
