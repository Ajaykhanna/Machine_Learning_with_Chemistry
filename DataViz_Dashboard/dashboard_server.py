from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dashboard_common import EXPORT_DIR, STATIC_DIR, ensure_directories
from dashboard_data import get_store, parse_id_argument


def _is_client_disconnect(exc: BaseException) -> bool:
    return isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError))


def _safe_join(base: Path, relative: str) -> Path | None:
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def build_handler():
    store = get_store()

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "TrajectoryDashboard/1.0"

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError as exc:
                if not _is_client_disconnect(exc):
                    raise

        def _send_file(self, path: Path) -> None:
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found.")
                return

            mime_type, _ = mimetypes.guess_type(str(path))
            content = path.read_bytes()
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except OSError as exc:
                if not _is_client_disconnect(exc):
                    raise

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                route = parsed.path
                params = parse_qs(parsed.query)

                if route in {"/", "/index.html"}:
                    self._send_file(STATIC_DIR / "index_v2.html")
                    return

                if route.startswith("/static/"):
                    relative = route.replace("/static/", "", 1)
                    file_path = _safe_join(STATIC_DIR, relative)
                    if file_path is None:
                        self.send_error(HTTPStatus.FORBIDDEN, "Invalid path.")
                        return
                    self._send_file(file_path)
                    return

                if route.startswith("/exports/"):
                    relative = route.replace("/exports/", "", 1)
                    file_path = _safe_join(EXPORT_DIR, relative)
                    if file_path is None:
                        self.send_error(HTTPStatus.FORBIDDEN, "Invalid path.")
                        return
                    self._send_file(file_path)
                    return

                if route == "/api/health":
                    self._send_json({"ok": True})
                    return

                if route == "/api/metadata":
                    self._send_json(store.metadata_payload())
                    return

                if route == "/api/series":
                    metric = params.get("metric", ["energy"])[0]
                    ids = parse_id_argument(params.get("ids", [""])[0])
                    start = int(params.get("start", ["0"])[0])
                    end = int(params.get("end", [str(store.metadata["snapshot_count"])])[0])
                    width = int(params.get("width", ["1200"])[0])
                    payload = store.get_series_payload(metric=metric, ids=ids, start=start, end=end, width=width)
                    self._send_json(payload)
                    return

                if route == "/api/histogram":
                    ids = parse_id_argument(params.get("ids", [""])[0])
                    start = int(params.get("start", ["0"])[0])
                    end = int(params.get("end", [str(store.metadata["snapshot_count"])])[0])
                    bins = int(params.get("bins", [str(store.config["defaults"]["histogram_bins"])])[0])
                    payload = store.get_histogram_payload(ids=ids, start=start, end=end, bins=bins)
                    self._send_json(payload)
                    return

                if route == "/api/statistics":
                    state_ids = parse_id_argument(params.get("states", [""])[0])
                    pair_ids = parse_id_argument(params.get("pairs", [""])[0])
                    start = int(params.get("start", ["0"])[0])
                    end = int(params.get("end", [str(store.metadata["snapshot_count"])])[0])
                    payload = store.get_statistics_payload(start=start, end=end, state_ids=state_ids, pair_ids=pair_ids)
                    self._send_json(payload)
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Route not found.")
            except Exception as exc:  # noqa: BLE001
                if _is_client_disconnect(exc):
                    return
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/export":
                self.send_error(HTTPStatus.NOT_FOUND, "Route not found.")
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))

            try:
                response = store.export_plot(payload)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self._send_json(response)

    return DashboardHandler


def run_server(host: str, port: int) -> None:
    ensure_directories()
    handler = build_handler()
    server = ThreadingHTTPServer((host, port), handler)

    print(f"Trajectory dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
    finally:
        server.server_close()
