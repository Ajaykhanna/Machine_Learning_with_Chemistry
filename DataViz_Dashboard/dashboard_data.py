from __future__ import annotations

import concurrent.futures
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib.format import open_memmap

import dashboard_common
from dashboard_common import load_or_init_config


STATE_PALETTE = [
    "#1768ac",
    "#f78154",
    "#1b998b",
    "#c5283d",
    "#6c5ce7",
    "#ff9f1c",
    "#386641",
    "#bc5090",
    "#2f4858",
    "#8ac926",
    "#9d4edd",
]

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "energy": {
        "kind": "state",
        "title": "Energies",
        "source": "acn_S.npy",
        "fallback_prefix": "acn_S",
        "y_label": "Energy",
    },
    "force_norm": {
        "kind": "state",
        "title": "Force Norms",
        "source": "acn_F.npy",
        "fallback_prefix": "acn_F",
        "y_label": "Force Norm",
    },
    "dipole_magnitude": {
        "kind": "state",
        "title": "Dipole Magnitudes",
        "source": "acn_D.npy",
        "fallback_prefix": "acn_D",
        "y_label": "Dipole Magnitude",
    },
    "nacr_norm": {
        "kind": "pair",
        "title": "NACR Norms",
        "source": "acn_NACR.npy",
        "fallback_prefix": "acn_NACR",
        "y_label": "NACR Norm",
    },
    "denacr_norm": {
        "kind": "pair",
        "title": "dENACR Norms",
        "source": "acn_dENACR.npy",
        "fallback_prefix": "acn_dENACR",
        "y_label": "dENACR Norm",
    },
}

METRIC_CHUNK_SIZES = {
    "energy": 262_144,
    "force_norm": 8_192,
    "dipole_magnitude": 65_536,
    "nacr_norm": 2_048,
    "denacr_norm": 4_096,
}

CACHE_BUILD_LOCK = threading.RLock()
MISSING_OVERRIDE_WARNINGS: set[str] = set()
MISSING_OVERRIDE_LOCK = threading.Lock()
CACHE_PROGRESS: "CacheProgress | None" = None


def active_metrics() -> tuple[str, ...]:
    return tuple(dashboard_common.ENABLED_METRICS)


def current_root() -> Path:
    return dashboard_common.ROOT


def current_cache_dir() -> Path:
    return dashboard_common.CACHE_DIR


def current_export_dir() -> Path:
    return dashboard_common.EXPORT_DIR


def current_cache_metadata_path() -> Path:
    return dashboard_common.CACHE_METADATA_PATH


def bracket_unit_label(raw_unit: str) -> str:
    return f"[{raw_unit}]" if raw_unit else ""


def axis_label_with_unit(label: str, raw_unit: str) -> str:
    suffix = bracket_unit_label(raw_unit)
    return f"{label} {suffix}".strip()


def metric_spec(metric: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_or_init_config()
    spec = dict(METRIC_SPECS[metric])
    file_config = config.get("files", {}).get(metric, {})
    axis_label = config.get("axis_labels", {}).get(metric)
    spec["source_override"] = file_config.get("source")
    spec["fallback_prefix_override"] = file_config.get("fallback_prefix")
    if axis_label:
        spec["y_label"] = str(axis_label)
    return spec


def _resolve_dataset_path(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = current_root() / path
    return path.resolve()


def warn_missing_override_once(cache_key: str, message: str) -> None:
    with MISSING_OVERRIDE_LOCK:
        if cache_key in MISSING_OVERRIDE_WARNINGS:
            return
        MISSING_OVERRIDE_WARNINGS.add(cache_key)
    print(f"Warning: {message}", file=sys.stderr)


def combined_source_candidates(metric: str, config: dict[str, Any] | None = None) -> list[tuple[Path, bool]]:
    spec = metric_spec(metric, config)
    default_path = current_root() / METRIC_SPECS[metric]["source"]
    candidates: list[tuple[Path, bool]] = []

    override = spec.get("source_override")
    if override:
        override_path = _resolve_dataset_path(str(override))
        candidates.append((override_path, True))
    if not candidates or candidates[0][0] != default_path:
        candidates.append((default_path, False))
    return candidates


def fallback_prefix_candidates(metric: str, config: dict[str, Any] | None = None) -> list[tuple[Path, bool]]:
    spec = metric_spec(metric, config)
    default_prefix = _resolve_dataset_path(METRIC_SPECS[metric]["fallback_prefix"])
    candidates: list[tuple[Path, bool]] = []

    override = spec.get("fallback_prefix_override")
    if override:
        override_prefix = _resolve_dataset_path(str(override))
        candidates.append((override_prefix, True))
    if not candidates or candidates[0][0] != default_prefix:
        candidates.append((default_prefix, False))
    return candidates


def fallback_file_candidates(metric: str, identifier: str | int, config: dict[str, Any] | None = None) -> list[tuple[Path, bool]]:
    formatted = str(identifier) if METRIC_SPECS[metric]["kind"] == "state" else format_pair_filename(str(identifier))
    candidates: list[tuple[Path, bool]] = []
    for prefix_path, is_override in fallback_prefix_candidates(metric, config):
        candidates.append((prefix_path.with_name(f"{prefix_path.name}{formatted}.npy"), is_override))
    return candidates


def resolve_existing_fallback_file(metric: str, identifier: str | int, config: dict[str, Any] | None = None) -> Path:
    for candidate, is_override in fallback_file_candidates(metric, identifier, config):
        if candidate.exists():
            return candidate
        if is_override:
            warn_missing_override_once(
                f"{metric}:fallback:{candidate}",
                f"Configured fallback prefix for {metric} was not found at '{candidate}'. Falling back to default naming.",
            )
    return fallback_file_candidates(metric, identifier, config)[-1][0]


def source_name_for_metric(metric: str, config: dict[str, Any] | None = None) -> str:
    for candidate, is_override in combined_source_candidates(metric, config):
        if candidate.exists():
            return candidate.name
        if is_override:
            warn_missing_override_once(
                f"{metric}:source:{candidate}",
                f"Configured source file for {metric} was not found at '{candidate}'. Falling back to default naming.",
            )
    return METRIC_SPECS[metric]["source"]


def _load_existing_source_path(metric: str, config: dict[str, Any] | None = None) -> Path | None:
    for candidate, is_override in combined_source_candidates(metric, config):
        if candidate.exists():
            return candidate
        if is_override:
            warn_missing_override_once(
                f"{metric}:source:{candidate}",
                f"Configured source file for {metric} was not found at '{candidate}'. Falling back to default naming.",
            )
    return None


def estimate_metric_work_units(metric: str, state_ids: list[int], pair_labels: list[str], config: dict[str, Any]) -> int:
    combined = _load_combined_source(metric, config)
    if combined is not None:
        snapshot_count = int(combined.shape[0])
        return max(1, math.ceil(snapshot_count / METRIC_CHUNK_SIZES[metric]))

    spec = metric_spec(metric, config)
    identifiers = state_ids if spec["kind"] == "state" else pair_labels
    first_path = resolve_existing_fallback_file(metric, identifiers[0], config)
    first_array = np.load(first_path, mmap_mode="r")
    snapshot_count = int(first_array.shape[0])
    return max(1, len(identifiers) * math.ceil(snapshot_count / METRIC_CHUNK_SIZES[metric]))


def initial_snapshot_count(config: dict[str, Any]) -> int:
    combined = _load_combined_source("energy", config)
    if combined is not None:
        return int(combined.shape[0])

    state_ids = discover_state_ids(config)
    first_path = resolve_existing_fallback_file("energy", state_ids[0], config)
    first_array = np.load(first_path, mmap_mode="r")
    return int(first_array.shape[0])


class CacheProgress:
    def __init__(self, total_units: int) -> None:
        self.total_units = max(1, total_units)
        self.completed_units = 0
        self.started_at = time.perf_counter()
        self.last_render_at = 0.0
        self.lock = threading.Lock()
        self.render(force=True)

    def advance(self, metric: str, units: int = 1) -> None:
        with self.lock:
            self.completed_units = min(self.total_units, self.completed_units + units)
            now = time.perf_counter()
            if not (self.completed_units >= self.total_units or now - self.last_render_at >= 0.08):
                return
            self.last_render_at = now
            self.render(metric=metric, force=self.completed_units >= self.total_units)

    def render(self, metric: str = "", force: bool = False) -> None:
        fraction = self.completed_units / self.total_units
        width = 30
        filled = min(width, int(round(width * fraction)))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.perf_counter() - self.started_at
        suffix = f" | {metric}" if metric else ""
        print(
            f"\rCaching dashboard data [{bar}] {self.completed_units}/{self.total_units} "
            f"({fraction * 100:5.1f}%) | {elapsed:5.1f}s{suffix}",
            end="" if not force else "\n",
            flush=True,
        )


def advance_cache_progress(metric: str, units: int = 1) -> None:
    if CACHE_PROGRESS is not None:
        CACHE_PROGRESS.advance(metric, units)


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r1, g1, b1 = c, x, 0
    elif h < 120:
        r1, g1, b1 = x, c, 0
    elif h < 180:
        r1, g1, b1 = 0, c, x
    elif h < 240:
        r1, g1, b1 = 0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x

    r = round((r1 + m) * 255)
    g = round((g1 + m) * 255)
    b = round((b1 + m) * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_pair_palette(pair_labels: Iterable[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    golden = 137.508
    for index, label in enumerate(pair_labels):
        hue = (18 + index * golden) % 360
        colors[label] = _hsl_to_hex(hue, 0.62, 0.48)
    return colors


def metric_path(metric: str) -> Path:
    return current_cache_dir() / f"{metric}.npy"


def overview_path(metric: str) -> Path:
    return current_cache_dir() / f"{metric}_overview.npz"


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def format_pair_filename(label: str) -> str:
    left, right = label.split("-")
    return f"{left}{right}"


def discover_state_ids(config: dict[str, Any] | None = None) -> list[int]:
    combined = _load_combined_source("energy", config)
    if combined is not None:
        return list(range(int(combined.shape[1])))

    ids: list[int] = []
    for prefix_path, is_override in fallback_prefix_candidates("energy", config):
        parent = prefix_path.parent
        if not parent.exists():
            if is_override:
                warn_missing_override_once(
                    f"energy:prefix_dir:{prefix_path}",
                    f"Configured energy fallback prefix directory '{parent}' was not found. Falling back to default naming.",
                )
            continue

        state_pattern = re.compile(rf"{re.escape(prefix_path.name)}(\d+)\.npy$")
        matches = 0
        for path in parent.glob(f"{prefix_path.name}*.npy"):
            match = state_pattern.fullmatch(path.name)
            if match:
                ids.append(int(match.group(1)))
                matches += 1
        if is_override and matches == 0:
            warn_missing_override_once(
                f"energy:prefix_scan:{prefix_path}",
                f"Configured energy fallback prefix '{prefix_path.name}' was not found. Falling back to default naming.",
            )
    ids = sorted(set(ids))
    if not ids:
        raise FileNotFoundError("No state-resolved energy files were found.")
    return ids


def build_pair_labels(state_ids: list[int]) -> list[str]:
    excited = [state_id for state_id in state_ids if state_id >= 1]
    return [f"{left}-{right}" for left, right in combinations(excited, 2)]


def reduce_metric_chunk(metric: str, chunk: np.ndarray) -> np.ndarray:
    if metric == "energy":
        return np.asarray(chunk, dtype=np.float64)
    if metric == "force_norm":
        return np.sqrt(np.square(chunk, dtype=np.float64).sum(axis=(-1, -2)))
    if metric == "dipole_magnitude":
        return np.sqrt(np.square(chunk, dtype=np.float64).sum(axis=-1))
    if metric == "nacr_norm":
        return np.sqrt(np.square(chunk, dtype=np.float64).sum(axis=(-1, -2)))
    if metric == "denacr_norm":
        return np.sqrt(np.square(chunk, dtype=np.float64).sum(axis=-1))
    raise KeyError(f"Unsupported metric: {metric}")


def minmax_downsample(values: np.ndarray, start: int, end: int, target_points: int) -> tuple[np.ndarray, np.ndarray]:
    if target_points < 4:
        target_points = 4

    series = np.asarray(values[start:end], dtype=np.float64)
    length = int(series.shape[0])
    if length <= target_points:
        xs = np.arange(start, end, dtype=np.int64)
        return xs, series

    bucket_count = max(1, target_points // 2)
    bucket_size = math.ceil(length / bucket_count)
    xs = np.empty(bucket_count * 2, dtype=np.int64)
    ys = np.empty(bucket_count * 2, dtype=np.float64)

    write_index = 0
    for bucket_index in range(bucket_count):
        local_start = bucket_index * bucket_size
        local_end = min(local_start + bucket_size, length)
        if local_start >= local_end:
            break
        chunk = series[local_start:local_end]
        min_index = int(np.argmin(chunk))
        max_index = int(np.argmax(chunk))
        first_offset = local_start + min_index
        second_offset = local_start + max_index

        if first_offset > second_offset:
            min_index, max_index = max_index, min_index
            first_offset, second_offset = second_offset, first_offset

        xs[write_index] = start + first_offset
        ys[write_index] = float(chunk[min_index])
        write_index += 1

        xs[write_index] = start + second_offset
        ys[write_index] = float(chunk[max_index])
        write_index += 1

    return xs[:write_index], ys[:write_index]


def build_overview_for_metric(metric: str, levels: list[int]) -> None:
    data = np.load(metric_path(metric), mmap_mode="r")
    _, series_count = data.shape
    payload: dict[str, np.ndarray] = {}

    for level in levels:
        if int(data.shape[0]) <= level:
            continue

        x_pack = np.full((series_count, level), -1, dtype=np.int64)
        y_pack = np.full((series_count, level), np.nan, dtype=np.float64)
        lengths = np.zeros(series_count, dtype=np.int32)

        for series_index in range(series_count):
            xs, ys = minmax_downsample(data[:, series_index], 0, int(data.shape[0]), level)
            count = min(level, int(xs.shape[0]))
            x_pack[series_index, :count] = xs[:count]
            y_pack[series_index, :count] = ys[:count]
            lengths[series_index] = count

        payload[f"x_{level}"] = x_pack
        payload[f"y_{level}"] = y_pack
        payload[f"len_{level}"] = lengths

    np.savez_compressed(overview_path(metric), **payload)


def _load_combined_source(metric: str, config: dict[str, Any] | None = None) -> np.ndarray | None:
    for source_path, is_override in combined_source_candidates(metric, config):
        if source_path.exists():
            return np.load(source_path, mmap_mode="r")
        if is_override:
            warn_missing_override_once(
                f"{metric}:source:{source_path}",
                f"Configured source file for {metric} was not found at '{source_path}'. Falling back to default naming.",
            )
    return None


def _compute_metric_task(
    metric: str,
    state_ids: list[int],
    pair_labels: list[str],
    levels: list[int],
) -> dict[str, Any]:
    config = load_or_init_config()
    spec = metric_spec(metric, config)
    output = metric_path(metric)
    combined = _load_combined_source(metric, config)

    if combined is not None:
        snapshot_count = int(combined.shape[0])
        series_count = int(combined.shape[1]) if combined.ndim > 1 else 1
        result = open_memmap(output, mode="w+", dtype=np.float64, shape=(snapshot_count, series_count))
        mins = np.full(series_count, np.inf, dtype=np.float64)
        maxs = np.full(series_count, -np.inf, dtype=np.float64)

        chunk_size = METRIC_CHUNK_SIZES[metric]
        for start in range(0, snapshot_count, chunk_size):
            end = min(start + chunk_size, snapshot_count)
            reduced = reduce_metric_chunk(metric, np.asarray(combined[start:end]))
            result[start:end] = reduced
            mins = np.minimum(mins, np.min(reduced, axis=0))
            maxs = np.maximum(maxs, np.max(reduced, axis=0))
            advance_cache_progress(metric)
    else:
        identifiers = state_ids if spec["kind"] == "state" else pair_labels
        first_array = np.load(resolve_existing_fallback_file(metric, identifiers[0], config), mmap_mode="r")
        snapshot_count = int(first_array.shape[0])
        series_count = len(identifiers)
        result = open_memmap(output, mode="w+", dtype=np.float64, shape=(snapshot_count, series_count))
        mins = np.full(series_count, np.inf, dtype=np.float64)
        maxs = np.full(series_count, -np.inf, dtype=np.float64)

        chunk_size = METRIC_CHUNK_SIZES[metric]
        for column, identifier in enumerate(identifiers):
            source = np.load(resolve_existing_fallback_file(metric, identifier, config), mmap_mode="r")
            for start in range(0, snapshot_count, chunk_size):
                end = min(start + chunk_size, snapshot_count)
                reduced = reduce_metric_chunk(metric, np.asarray(source[start:end]))
                result[start:end, column] = reduced
                mins[column] = min(mins[column], float(np.min(reduced)))
                maxs[column] = max(maxs[column], float(np.max(reduced)))
                advance_cache_progress(metric)

    result.flush()
    build_overview_for_metric(metric, levels)

    return {
        "metric": metric,
        "shape": [int(result.shape[0]), int(result.shape[1])],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "metric_path": output.name,
        "overview_path": overview_path(metric).name,
    }


def metadata_shape_metrics(metric_results: list[dict[str, Any]]) -> tuple[str, ...]:
    ordered: list[str] = ["energy"]
    for item in metric_results:
        metric = item["metric"]
        if metric not in ordered:
            ordered.append(metric)
    return tuple(ordered)


def raw_shape_summary(
    state_ids: list[int],
    pair_labels: list[str],
    config: dict[str, Any],
    metrics_to_inspect: tuple[str, ...],
) -> dict[str, list[int]]:
    summary: dict[str, list[int]] = {}
    for metric in metrics_to_inspect:
        spec = metric_spec(metric, config)
        combined = _load_combined_source(metric, config)
        if combined is not None:
            raw = combined
            summary[metric] = [int(value) for value in raw.shape]
            continue

        if spec["kind"] == "state":
            sample = np.load(resolve_existing_fallback_file(metric, state_ids[0], config), mmap_mode="r")
            summary[metric] = [int(sample.shape[0]), len(state_ids), *[int(v) for v in sample.shape[1:]]]
        else:
            sample = np.load(resolve_existing_fallback_file(metric, pair_labels[0], config), mmap_mode="r")
            summary[metric] = [int(sample.shape[0]), len(pair_labels), *[int(v) for v in sample.shape[1:]]]
    return summary


def build_cache_metadata(metric_results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    state_ids = discover_state_ids(config)
    pair_labels = build_pair_labels(state_ids)
    metrics_to_inspect = metadata_shape_metrics(metric_results)
    raw_shapes = raw_shape_summary(state_ids, pair_labels, config, metrics_to_inspect)
    colors = {
        "states": {f"S{state_id}": STATE_PALETTE[index % len(STATE_PALETTE)] for index, state_id in enumerate(state_ids)},
        "pairs": build_pair_palette(pair_labels),
    }

    metadata = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "state_ids": state_ids,
        "state_labels": [f"S{state_id}" for state_id in state_ids],
        "pair_labels": pair_labels,
        "default_states": config["defaults"]["states"],
        "default_pairs": config["defaults"]["pairs"],
        "overview_levels": config["overview_levels"],
        "enabled_metrics": list(active_metrics()),
        "units": config["units"],
        "colors": colors,
        "raw_shapes": raw_shapes,
        "snapshot_count": raw_shapes["energy"][0],
        "metrics": {item["metric"]: item for item in metric_results},
        "sources": {
            source_name_for_metric(metric, config): source_signature(_load_existing_source_path(metric, config))
            for metric in metrics_to_inspect
            if _load_existing_source_path(metric, config) is not None
        },
    }
    return metadata


def rebuild_cache(rebuild: bool = False, workers: int | None = None) -> dict[str, Any]:
    global CACHE_PROGRESS
    with CACHE_BUILD_LOCK:
        config = load_or_init_config()
        state_ids = discover_state_ids(config)
        pair_labels = build_pair_labels(state_ids)
        enabled = active_metrics()
        metadata_path = current_cache_metadata_path()

        required = [metric_path(metric) for metric in enabled]
        if not rebuild and metadata_path.exists() and all(path.exists() for path in required):
            with metadata_path.open("r", encoding="utf-8") as handle:
                cached_metadata = json.load(handle)

            if "metrics" in cached_metadata and all(metric in cached_metadata["metrics"] for metric in enabled):
                metric_results = [cached_metadata["metrics"][metric] for metric in enabled]
                refreshed_metadata = build_cache_metadata(metric_results, config)
                if refreshed_metadata != cached_metadata:
                    metadata_path.write_text(json.dumps(refreshed_metadata, indent=2), encoding="utf-8")
                return refreshed_metadata

        levels = [int(level) for level in config["overview_levels"]]
        max_workers = workers or min(len(enabled), 4)
        max_workers = max(1, min(max_workers, len(enabled)))

        snapshot_count = initial_snapshot_count(config)
        total_work_units = sum(estimate_metric_work_units(metric, state_ids, pair_labels, config) for metric in enabled)
        CACHE_PROGRESS = CacheProgress(total_work_units) if snapshot_count > 50_000 else None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    executor.submit(_compute_metric_task, metric, state_ids, pair_labels, levels)
                    for metric in enabled
                ]
                results = [task.result() for task in concurrent.futures.as_completed(tasks)]
        finally:
            if CACHE_PROGRESS is not None and CACHE_PROGRESS.completed_units < CACHE_PROGRESS.total_units:
                CACHE_PROGRESS.render(metric="done", force=True)
            CACHE_PROGRESS = None

        results.sort(key=lambda item: list(METRIC_SPECS.keys()).index(item["metric"]))
        metadata = build_cache_metadata(results, config)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata


def _closest_overview_level(levels: list[int], requested_points: int) -> int | None:
    eligible = [level for level in levels if level >= requested_points]
    if eligible:
        return min(eligible)
    return max(levels) if levels else None


@dataclass
class DashboardStore:
    root: Path | None = None

    def __post_init__(self) -> None:
        if self.root is None:
            self.root = current_root()
        self.config = load_or_init_config()
        self.metadata = rebuild_cache(rebuild=False)
        self._metric_arrays: dict[str, np.ndarray] = {}
        self._overviews: dict[str, dict[str, np.ndarray]] = {}
        self._ensure_required_metric_files()

    def refresh(self, rebuild: bool = False) -> None:
        self.config = load_or_init_config()
        self.metadata = rebuild_cache(rebuild=rebuild)
        self._metric_arrays.clear()
        self._overviews.clear()
        self._ensure_required_metric_files()

    def ensure_metric_enabled(self, metric: str) -> None:
        if metric not in active_metrics():
            raise ValueError(f"Metric '{metric}' is not enabled for this dashboard run.")

    def _ensure_required_metric_files(self) -> None:
        missing = [metric for metric in active_metrics() if not metric_path(metric).exists()]
        if missing:
            self.metadata = rebuild_cache(rebuild=True)

    def _load_metric_array(self, metric: str) -> np.ndarray:
        try:
            return np.load(metric_path(metric), mmap_mode="r")
        except (FileNotFoundError, OSError):
            self.refresh(rebuild=True)
            return np.load(metric_path(metric), mmap_mode="r")

    def get_metric_array(self, metric: str) -> np.ndarray:
        self.ensure_metric_enabled(metric)
        if metric not in self._metric_arrays:
            self._metric_arrays[metric] = self._load_metric_array(metric)
        return self._metric_arrays[metric]

    def get_overview_pack(self, metric: str) -> dict[str, np.ndarray]:
        self.ensure_metric_enabled(metric)
        if metric not in self._overviews:
            pack_path = overview_path(metric)
            if pack_path.exists():
                with np.load(pack_path, allow_pickle=False) as archive:
                    self._overviews[metric] = {key: archive[key] for key in archive.files}
            else:
                self._overviews[metric] = {}
        return self._overviews[metric]

    def resolve_indices(self, metric: str, ids: list[str] | list[int] | None) -> tuple[list[int], list[str], dict[str, str]]:
        self.ensure_metric_enabled(metric)
        spec = metric_spec(metric, self.config)
        if spec["kind"] == "state":
            all_ids = self.metadata["state_ids"]
            if not ids:
                ids = self.metadata["default_states"]
            normalized = [int(str(value).replace("S", "")) for value in ids]
            indices = [all_ids.index(value) for value in normalized if value in all_ids]
            resolved_labels = [f"S{value}" for value in normalized if value in all_ids]
            return indices, resolved_labels, self.metadata["colors"]["states"]

        all_pairs = self.metadata["pair_labels"]
        if not ids:
            ids = self.metadata["default_pairs"]
        normalized_pairs = [str(value).replace("_", "-") for value in ids]
        indices = [all_pairs.index(value) for value in normalized_pairs if value in all_pairs]
        return indices, [value for value in normalized_pairs if value in all_pairs], self.metadata["colors"]["pairs"]

    def get_series_payload(
        self,
        metric: str,
        ids: list[str] | list[int] | None,
        start: int,
        end: int,
        width: int,
    ) -> dict[str, Any]:
        data = self.get_metric_array(metric)
        snapshot_count = int(data.shape[0])
        start = max(0, min(start, snapshot_count - 1))
        end = max(start + 1, min(end, snapshot_count))
        width = max(320, int(width))

        indices, labels, colors = self.resolve_indices(metric, ids)
        max_points = min(max(512, width * 2), 4096)

        use_overview = start == 0 and end == snapshot_count
        overview_level = None
        overview_pack: dict[str, np.ndarray] | None = None
        if use_overview:
            level = _closest_overview_level(self.metadata.get("overview_levels", []), max_points)
            if level is not None:
                pack = self.get_overview_pack(metric)
                if f"x_{level}" in pack:
                    overview_level = level
                    overview_pack = pack

        series_payload = []
        for column_index, label in zip(indices, labels, strict=True):
            if overview_level is not None and overview_pack is not None:
                valid_count = int(overview_pack[f"len_{overview_level}"][column_index])
                xs = overview_pack[f"x_{overview_level}"][column_index][:valid_count]
                ys = overview_pack[f"y_{overview_level}"][column_index][:valid_count]
            else:
                xs, ys = minmax_downsample(data[:, column_index], start, end, max_points)

            series_payload.append(
                {
                    "id": label,
                    "label": label,
                    "color": colors[label],
                    "x": xs.astype(np.int64).tolist(),
                    "y": np.round(ys.astype(np.float64), 8).tolist(),
                }
            )

        return {
            "metric": metric,
            "title": METRIC_SPECS[metric]["title"],
            "axis_label": self.config["axis_labels"][metric],
            "units": self.metadata["units"][metric],
            "start": start,
            "end": end,
            "snapshot_count": snapshot_count,
            "series": series_payload,
        }

    def get_histogram_payload(
        self,
        ids: list[str] | list[int] | None,
        start: int,
        end: int,
        bins: int,
    ) -> dict[str, Any]:
        self.ensure_metric_enabled("energy")
        data = self.get_metric_array("energy")
        snapshot_count = int(data.shape[0])
        start = max(0, min(start, snapshot_count - 1))
        end = max(start + 1, min(end, snapshot_count))
        bins = max(12, min(int(bins), 240))

        indices, labels, colors = self.resolve_indices("energy", ids)
        histogram_payload = []
        for column_index, label in zip(indices, labels, strict=True):
            values = np.asarray(data[start:end, column_index], dtype=np.float64)
            counts, edges = np.histogram(values, bins=bins)
            histogram_payload.append(
                {
                    "id": label,
                    "label": label,
                    "color": colors[label],
                    "counts": counts.astype(np.int64).tolist(),
                    "edges": np.round(edges.astype(np.float64), 8).tolist(),
                }
            )

        return {
            "metric": "energy_histogram",
            "title": "Energy Distribution",
            "axis_label": self.config["axis_labels"]["energy"],
            "units": self.metadata["units"]["energy"],
            "start": start,
            "end": end,
            "bins": bins,
            "snapshot_count": snapshot_count,
            "series": histogram_payload,
        }

    def get_statistics_payload(
        self,
        start: int,
        end: int,
        state_ids: list[str] | list[int] | None,
        pair_ids: list[str] | list[int] | None,
    ) -> dict[str, Any]:
        snapshot_count = int(self.metadata["snapshot_count"])
        start = max(0, min(start, snapshot_count - 1))
        end = max(start + 1, min(end, snapshot_count))

        payload: dict[str, Any] = {
            "start": start,
            "end": end,
            "snapshot_count": snapshot_count,
            "metrics": {},
        }

        for metric in active_metrics():
            spec = metric_spec(metric, self.config)
            ids = state_ids if spec["kind"] == "state" else pair_ids
            indices, labels, _ = self.resolve_indices(metric, ids)
            data = self.get_metric_array(metric)

            summaries: list[dict[str, Any]] = []
            total_count = 0
            total_sum = 0.0
            total_sum_sq = 0.0
            global_min = np.inf
            global_max = -np.inf

            for column_index, label in zip(indices, labels, strict=True):
                series_values = np.asarray(data[start:end, column_index], dtype=np.float64)
                series_sum = float(np.sum(series_values))
                series_count = int(series_values.shape[0])
                series_mean = series_sum / series_count
                centered = series_values - series_mean
                series_var = float(np.mean(np.square(centered)))
                series_std = math.sqrt(series_var)
                series_min = float(np.min(series_values))
                series_max = float(np.max(series_values))

                summaries.append(
                    {
                        "id": label,
                        "mean": series_mean,
                        "std": series_std,
                        "variance": series_var,
                        "min": series_min,
                        "max": series_max,
                        "count": series_count,
                    }
                )

                total_count += series_count
                total_sum += series_sum
                total_sum_sq += float(np.sum(np.square(series_values)))
                global_min = min(global_min, series_min)
                global_max = max(global_max, series_max)

            aggregate_mean = total_sum / total_count if total_count else 0.0
            aggregate_var = max((total_sum_sq / total_count) - aggregate_mean**2, 0.0) if total_count else 0.0
            aggregate_std = math.sqrt(aggregate_var)

            payload["metrics"][metric] = {
                "title": spec["title"],
                "axis_label": self.config["axis_labels"][metric],
                "units": self.metadata["units"][metric],
                "kind": spec["kind"],
                "selected_series_count": len(summaries),
                "series": summaries,
                "aggregate": {
                    "mean": aggregate_mean,
                    "std": aggregate_std,
                    "variance": aggregate_var,
                    "min": float(global_min) if total_count else 0.0,
                    "max": float(global_max) if total_count else 0.0,
                    "count": total_count,
                },
            }

        return payload

    def export_plot(self, payload: dict[str, Any]) -> dict[str, Any]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metric = str(payload.get("metric", "energy"))
        self.ensure_metric_enabled(metric)
        export_format = str(payload.get("format", "png")).lower()
        plot_type = str(payload.get("plot_type", "line"))
        ids = payload.get("ids")
        start = int(payload.get("start", 0))
        end = int(payload.get("end", self.metadata["snapshot_count"]))
        bins = int(payload.get("bins", self.config["defaults"]["histogram_bins"]))
        dpi = int(payload.get("dpi", self.config["export"]["dpi"]))
        figure_size = tuple(self.config["export"]["figure_size"])
        allowed_formats = dashboard_common.normalize_export_formats(self.config["export"].get("formats"))

        if export_format not in {"png", "pdf"}:
            raise ValueError("Export format must be png or pdf.")
        if export_format not in allowed_formats:
            raise ValueError(f"Export format '{export_format}' is disabled by the current dashboard configuration.")

        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#ffffff")

        if plot_type == "histogram":
            histogram = self.get_histogram_payload(ids=ids, start=start, end=end, bins=bins)
            for series in histogram["series"]:
                edges = np.asarray(series["edges"], dtype=np.float64)
                counts = np.asarray(series["counts"], dtype=np.int64)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, counts, where="mid", color=series["color"], linewidth=1.7, label=series["label"])

            ax.set_title("Energy Distribution")
            ax.set_ylabel("Count")
            ax.set_xlabel(axis_label_with_unit("Energy", self.metadata["units"]["energy"]))
        else:
            line_data = self.get_series_payload(metric=metric, ids=ids, start=start, end=end, width=2400)
            for series in line_data["series"]:
                ax.plot(
                    np.asarray(series["x"], dtype=np.int64),
                    np.asarray(series["y"], dtype=np.float64),
                    color=series["color"],
                    linewidth=self.config["export"]["line_width"],
                    label=series["label"],
                )

            spec = metric_spec(metric, self.config)
            ax.set_title(METRIC_SPECS[metric]["title"])
            ax.set_ylabel(axis_label_with_unit(spec["y_label"], self.metadata["units"][metric]))
            ax.set_xlabel("Snapshot index")

        ax.grid(False)
        ax.legend(frameon=False, loc="best", ncol=min(4, max(1, len(ax.lines) or len(ax.patches))))
        fig.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metric}_{plot_type}_{timestamp}.{export_format}"
        file_path = current_export_dir() / filename
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        return {
            "ok": True,
            "metric": metric,
            "plot_type": plot_type,
            "format": export_format,
            "file": str(file_path.resolve()),
            "filename": filename,
        }

    def metadata_payload(self) -> dict[str, Any]:
        raw_shapes = self.metadata["raw_shapes"]
        raw_shape_payload: dict[str, list[int]] = {}
        if "energy" in raw_shapes:
            raw_shape_payload["S"] = raw_shapes["energy"]
        if "force_norm" in raw_shapes:
            raw_shape_payload["F"] = raw_shapes["force_norm"]
        if "dipole_magnitude" in raw_shapes:
            raw_shape_payload["D"] = raw_shapes["dipole_magnitude"]
        if "nacr_norm" in raw_shapes:
            raw_shape_payload["NACR"] = raw_shapes["nacr_norm"]
        if "denacr_norm" in raw_shapes:
            raw_shape_payload["dENACR"] = raw_shapes["denacr_norm"]
        return {
            "snapshot_count": self.metadata["snapshot_count"],
            "state_ids": self.metadata["state_ids"],
            "state_labels": self.metadata["state_labels"],
            "pair_labels": self.metadata["pair_labels"],
            "default_states": self.metadata["default_states"],
            "default_pairs": self.metadata["default_pairs"],
            "overview_levels": self.metadata["overview_levels"],
            "enabled_metrics": self.metadata.get("enabled_metrics", list(active_metrics())),
            "units": self.metadata["units"],
            "axis_labels": self.config["axis_labels"],
            "export_formats": list(dashboard_common.normalize_export_formats(self.config["export"].get("formats"))),
            "colors": self.metadata["colors"],
            "raw_shapes": raw_shape_payload,
            "metric_titles": {metric: spec["title"] for metric, spec in METRIC_SPECS.items() if metric in active_metrics()},
            "generated_at": self.metadata["generated_at"],
        }


def parse_id_argument(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_store() -> DashboardStore:
    return DashboardStore()
