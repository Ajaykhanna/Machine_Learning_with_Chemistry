from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from dashboard_common import add_runtime_cli_arguments, configure_runtime, maybe_reexec_in_ml_env, resolve_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the offline trajectory dashboard.")
    parser.add_argument("--host", default=None, help="Interface to bind the HTTP server to.")
    parser.add_argument("--port", type=int, default=None, help="HTTP port to use for the dashboard.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the dashboard in a browser tab.",
    )
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

    from dashboard_server import run_server

    host = settings["host"]
    port = settings["port"] if settings["port"] is not None else 8127
    url = f"http://{host}:{port}"

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    run_server(host, port)


if __name__ == "__main__":
    main()
