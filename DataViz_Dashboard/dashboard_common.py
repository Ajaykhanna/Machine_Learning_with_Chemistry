from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
ALL_METRICS = ("energy", "force_norm", "dipole_magnitude", "nacr_norm", "denacr_norm")
DEFAULT_ENABLED_METRICS = ("energy",)
DEFAULT_EXPORT_FORMATS = ("png", "pdf")
METRIC_FLAG_MAP = {
    "energies": "energy",
    "energy": "energy",
    "forces_norm": "force_norm",
    "force_norm": "force_norm",
    "dipole_magnitude": "dipole_magnitude",
    "dipoles": "dipole_magnitude",
    "nacrs_norm": "nacr_norm",
    "nacr_norm": "nacr_norm",
    "scaled_nacrs": "denacr_norm",
    "scaled_nacr": "denacr_norm",
    "denacr_norm": "denacr_norm",
}

METRIC_INPUT_ALIASES = {
    "energy": ("energy", "energies"),
    "force_norm": ("force", "forces", "force_norm", "forces_norm"),
    "dipole_magnitude": ("dipole", "dipoles", "dipole_magnitude"),
    "nacr_norm": ("nacr", "nacrs", "nacr_norm", "nacrs_norm"),
    "denacr_norm": ("denacr", "denacrs", "scaled_nacr", "scaled_nacrs", "denacr_norm"),
}

ROOT = SCRIPT_ROOT
STATIC_DIR = SCRIPT_ROOT / "dashboard_static"
CACHE_DIR = ROOT / "dashboard_cache"
EXPORT_DIR = ROOT / "exports"
CONFIG_PATH = ROOT / "dashboard_config.json"
CACHE_METADATA_PATH = CACHE_DIR / "metadata.json"
ENABLED_METRICS = DEFAULT_ENABLED_METRICS
RUNTIME_CONFIG_OVERRIDES: dict[str, Any] = {}

DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {
        "states": [0, 1, 2, 3],
        "pairs": ["1-2", "1-3", "2-3"],
        "histogram_bins": 64,
    },
    "units": {
        "energy": "eV",
        "force_norm": "eV/$\\AA$",
        "dipole_magnitude": "atomic units",
        "nacr_norm": "1/$\\AA$",
        "denacr_norm": "eV/$\\AA$",
    },
    "axis_labels": {
        "energy": "Energy",
        "force_norm": "Force Norm",
        "dipole_magnitude": "Dipole Magnitude",
        "nacr_norm": "NACR Norm",
        "denacr_norm": "dENACR Norm",
    },
    "files": {
        "energy": {"source": "acn_S.npy", "fallback_prefix": "acn_S"},
        "force_norm": {"source": "acn_F.npy", "fallback_prefix": "acn_F"},
        "dipole_magnitude": {"source": "acn_D.npy", "fallback_prefix": "acn_D"},
        "nacr_norm": {"source": "acn_NACR.npy", "fallback_prefix": "acn_NACR"},
        "denacr_norm": {"source": "acn_dENACR.npy", "fallback_prefix": "acn_dENACR"},
    },
    "export": {
        "dpi": 300,
        "figure_size": [10.5, 5.8],
        "line_width": 1.8,
        "formats": list(DEFAULT_EXPORT_FORMATS),
    },
    "overview_levels": [512, 1024, 2048, 4096],
}

INPUT_SOURCE_KEY_MAP: dict[str, str] = {}
INPUT_PREFIX_KEY_MAP: dict[str, str] = {}
INPUT_AXIS_KEY_MAP: dict[str, str] = {}
for metric, aliases in METRIC_INPUT_ALIASES.items():
    for alias in aliases:
        INPUT_SOURCE_KEY_MAP[f"{alias}_file"] = metric
        INPUT_PREFIX_KEY_MAP[f"{alias}_prefix"] = metric
        INPUT_AXIS_KEY_MAP[f"{alias}_axis_label"] = metric


def ensure_directories() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def normalize_metric_names(metrics: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not metrics:
        return DEFAULT_ENABLED_METRICS

    normalized: list[str] = []
    for metric in metrics:
        canonical = METRIC_FLAG_MAP.get(str(metric).strip().lower(), str(metric).strip().lower())
        if canonical == "all":
            return tuple(ALL_METRICS)
        if canonical not in ALL_METRICS:
            raise ValueError(f"Unsupported dashboard metric: {metric}")
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized or DEFAULT_ENABLED_METRICS)


def normalize_export_formats(formats: list[str] | tuple[str, ...] | str | None) -> tuple[str, ...]:
    if not formats:
        return DEFAULT_EXPORT_FORMATS

    if isinstance(formats, str):
        raw_values = [item.strip().lower() for item in formats.split(",") if item.strip()]
    else:
        raw_values = [str(item).strip().lower() for item in formats if str(item).strip()]

    normalized: list[str] = []
    for value in raw_values:
        if value in {"both", "all"}:
            return DEFAULT_EXPORT_FORMATS
        if value not in {"png", "pdf"}:
            raise ValueError(f"Unsupported export format: {value}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized or DEFAULT_EXPORT_FORMATS)


def configure_runtime(
    data_dir: str | Path | None = None,
    enabled_metrics: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> None:
    global ROOT, CACHE_DIR, EXPORT_DIR, CONFIG_PATH, CACHE_METADATA_PATH, ENABLED_METRICS, RUNTIME_CONFIG_OVERRIDES

    resolved_root = Path(data_dir).expanduser().resolve() if data_dir else SCRIPT_ROOT
    resolved_output = Path(output_dir).expanduser() if output_dir else (resolved_root / "exports")
    if not resolved_output.is_absolute():
        resolved_output = (resolved_root / resolved_output).resolve()
    else:
        resolved_output = resolved_output.resolve()

    ROOT = resolved_root
    CACHE_DIR = ROOT / "dashboard_cache"
    EXPORT_DIR = resolved_output
    CONFIG_PATH = ROOT / "dashboard_config.json"
    CACHE_METADATA_PATH = CACHE_DIR / "metadata.json"
    ENABLED_METRICS = normalize_metric_names(enabled_metrics)
    RUNTIME_CONFIG_OVERRIDES = runtime_overrides or {}
    ensure_directories()


def add_runtime_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input_file",
        "--input-file",
        dest="input_file",
        default=None,
        help="Optional dashboard input file with data paths, filenames, export settings, and metric selections.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing the trajectory .npy files and dashboard cache/output folders.",
    )
    parser.add_argument("--all", action="store_true", help="Enable all dashboard metrics.")
    parser.add_argument("--energy", "--energies", dest="energies", action="store_true", help="Enable energy trajectories and distributions.")
    parser.add_argument("--forces", "--forces_norm", dest="forces_norm", action="store_true", help="Enable force norm panels.")
    parser.add_argument("--dipoles", "--dipole_magnitude", dest="dipole_magnitude", action="store_true", help="Enable dipole magnitude panels.")
    parser.add_argument("--nacrs", "--nacrs_norm", dest="nacrs_norm", action="store_true", help="Enable NACR norm panels.")
    parser.add_argument(
        "--scaled_nacrs",
        "--scaled-nacrs",
        dest="scaled_nacrs",
        action="store_true",
        help="Enable dENACR norm panels.",
    )
    return parser


def metric_flags_present(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, attribute, False)
        for attribute in ("all", "energies", "forces_norm", "dipole_magnitude", "nacrs_norm", "scaled_nacrs")
    )


def resolve_enabled_metrics_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "all", False):
        return tuple(ALL_METRICS)

    selected: list[str] = []
    if getattr(args, "energies", False):
        selected.append("energy")
    if getattr(args, "forces_norm", False):
        selected.append("force_norm")
    if getattr(args, "dipole_magnitude", False):
        selected.append("dipole_magnitude")
    if getattr(args, "nacrs_norm", False):
        selected.append("nacr_norm")
    if getattr(args, "scaled_nacrs", False):
        selected.append("denacr_norm")

    return normalize_metric_names(selected or ["energy"])


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value) if key in base else value
        return merged
    return override if override is not None else base


def _normalize_input_key(raw_key: str) -> str:
    return raw_key.strip().lower().replace("-", "_").replace(" ", "_")


def _parse_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_int(raw_value: str, key: str) -> int:
    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ValueError(f"Input file key '{key}' expects an integer value.") from exc


def parse_input_file(path: str | Path) -> dict[str, Any]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    parsed: dict[str, Any] = {}
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                raise ValueError(f"Invalid input file line {line_number}: expected 'key = value'.")

            key, value = stripped.split("=", 1)
            normalized_key = _normalize_input_key(key)
            parsed[normalized_key] = value.strip()

    return parsed


def build_runtime_overrides_from_input(raw_settings: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    file_overrides: dict[str, dict[str, str]] = {}
    axis_overrides: dict[str, str] = {}

    for key, metric in INPUT_SOURCE_KEY_MAP.items():
        value = raw_settings.get(key)
        if value:
            file_overrides.setdefault(metric, {})["source"] = str(value)

    for key, metric in INPUT_PREFIX_KEY_MAP.items():
        value = raw_settings.get(key)
        if value:
            file_overrides.setdefault(metric, {})["fallback_prefix"] = str(value)

    for key, metric in INPUT_AXIS_KEY_MAP.items():
        value = raw_settings.get(key)
        if value:
            axis_overrides[metric] = str(value)

    format_value = raw_settings.get("save_formats") or raw_settings.get("saving_formats") or raw_settings.get("export_formats")
    if format_value:
        overrides.setdefault("export", {})["formats"] = list(normalize_export_formats(format_value))

    if file_overrides:
        overrides["files"] = file_overrides
    if axis_overrides:
        overrides["axis_labels"] = axis_overrides

    return overrides


def _resolve_metrics_from_input(raw_settings: dict[str, Any]) -> tuple[str, ...] | None:
    metrics_value = raw_settings.get("metrics")
    if not metrics_value:
        return None
    return normalize_metric_names(_parse_csv(str(metrics_value)))


def _prefer_cli_value(cli_value: Any, cli_present: bool, input_value: Any, label: str) -> Any:
    if cli_present and input_value not in {None, ""}:
        _warn(f"Both --input_file and CLI {label} were provided. The CLI value takes priority.")
    return cli_value if cli_present else input_value


def resolve_runtime_settings(args: argparse.Namespace) -> dict[str, Any]:
    raw_settings: dict[str, Any] = {}
    if getattr(args, "input_file", None):
        raw_settings = parse_input_file(args.input_file)

    cli_has_data_dir = bool(getattr(args, "data_dir", None))
    cli_has_host = getattr(args, "host", None) is not None
    cli_has_port = getattr(args, "port", None) is not None
    cli_has_metrics = metric_flags_present(args)

    input_metrics = _resolve_metrics_from_input(raw_settings)
    if cli_has_metrics and input_metrics is not None:
        _warn("Both --input_file and CLI metric flags were provided. The CLI metric selection takes priority.")

    enabled_metrics = (
        resolve_enabled_metrics_from_args(args)
        if cli_has_metrics
        else (input_metrics if input_metrics is not None else DEFAULT_ENABLED_METRICS)
    )

    data_dir = _prefer_cli_value(getattr(args, "data_dir", None), cli_has_data_dir, raw_settings.get("data_dir"), "--data_dir")
    host = _prefer_cli_value(getattr(args, "host", None), cli_has_host, raw_settings.get("host"), "--host") or "127.0.0.1"

    input_port_raw = raw_settings.get("port") or raw_settings.get("ports")
    input_port = _parse_int(str(input_port_raw), "port") if input_port_raw else None
    port = _prefer_cli_value(getattr(args, "port", None), cli_has_port, input_port, "--port")

    return {
        "input_settings": raw_settings,
        "data_dir": data_dir,
        "output_dir": raw_settings.get("output_dir"),
        "host": host,
        "port": port,
        "enabled_metrics": enabled_metrics,
        "runtime_overrides": build_runtime_overrides_from_input(raw_settings),
    }


def load_or_init_config() -> dict[str, Any]:
    ensure_directories()
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        file_config = dict(DEFAULT_CONFIG)
    else:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)

        file_config = _deep_merge(DEFAULT_CONFIG, loaded)
        if file_config != loaded:
            CONFIG_PATH.write_text(json.dumps(file_config, indent=2), encoding="utf-8")

    return _deep_merge(file_config, RUNTIME_CONFIG_OVERRIDES)


def find_ml_env_python() -> Path | None:
    env_candidates: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix_path = Path(conda_prefix)
        if prefix_path.name.lower() == "ml_env":
            env_candidates.append(prefix_path / "python.exe")

    user_home = Path.home()
    env_candidates.extend(
        [
            user_home / "AppData" / "Local" / "anaconda3" / "envs" / "ml_env" / "python.exe",
            user_home / "anaconda3" / "envs" / "ml_env" / "python.exe",
            Path("C:/ProgramData/anaconda3/envs/ml_env/python.exe"),
            Path("C:/ProgramData/miniconda3/envs/ml_env/python.exe"),
        ]
    )

    for candidate in env_candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def maybe_reexec_in_ml_env(script_path: Path, argv: list[str]) -> None:
    target_python = find_ml_env_python()
    if target_python is None:
        return

    current_python = Path(sys.executable).resolve()
    if current_python == target_python:
        return

    if os.environ.get("MD_DASHBOARD_REEXEC") == "1":
        return

    environment = os.environ.copy()
    environment["MD_DASHBOARD_REEXEC"] = "1"
    result = subprocess.run(
        [str(target_python), str(script_path), *argv],
        env=environment,
        check=False,
    )
    raise SystemExit(result.returncode)
