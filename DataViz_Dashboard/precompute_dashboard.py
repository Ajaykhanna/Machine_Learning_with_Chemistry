from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dashboard_common import add_runtime_cli_arguments, configure_runtime, maybe_reexec_in_ml_env, resolve_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute cache files for the trajectory dashboard.")
    parser.add_argument("--rebuild", action="store_true", help="Force a rebuild even if cache files already exist.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes to use.")
    add_runtime_cli_arguments(parser)
    return parser.parse_args()


def main() -> None:
    maybe_reexec_in_ml_env(Path(__file__), sys.argv[1:])
    args = parse_args()
    settings = resolve_runtime_settings(args)
    configure_runtime(
        data_dir=settings["data_dir"],
        enabled_metrics=settings["enabled_metrics"],
        output_dir=settings["output_dir"],
        runtime_overrides=settings["runtime_overrides"],
    )

    from dashboard_data import rebuild_cache

    metadata = rebuild_cache(rebuild=args.rebuild, workers=args.workers)
    print("Dashboard cache ready.")
    print(f"Snapshots: {metadata['snapshot_count']}")
    print(f"States: {', '.join(metadata['state_labels'])}")
    print(f"Enabled metrics: {', '.join(metadata['enabled_metrics'])}")
    print(f"Default excited-state pairs: {', '.join(metadata['default_pairs'])}")


if __name__ == "__main__":
    main()
