from __future__ import annotations

import copy
import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

import dashboard_common
from dashboard_common import configure_runtime
configure_runtime(enabled_metrics=["energy", "force_norm", "dipole_magnitude", "nacr_norm", "denacr_norm"])

from dashboard_data import DashboardStore, build_pair_labels, discover_state_ids


class DashboardDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = DashboardStore()

    def test_metadata_shapes_match_dataset(self) -> None:
        metadata = self.store.metadata_payload()
        self.assertEqual(
            metadata["enabled_metrics"],
            ["energy", "force_norm", "dipole_magnitude", "nacr_norm", "denacr_norm"],
        )
        self.assertEqual(metadata["raw_shapes"]["S"], [50000, 11])
        self.assertEqual(metadata["raw_shapes"]["F"], [50000, 11, 15, 3])
        self.assertEqual(metadata["raw_shapes"]["D"], [50000, 11, 3])
        self.assertEqual(metadata["raw_shapes"]["NACR"], [50000, 45, 15, 3])
        self.assertEqual(metadata["raw_shapes"]["dENACR"], [50000, 45, 45])

    def test_force_norm_matches_direct_computation(self) -> None:
        raw = np.load("acn_F.npy", mmap_mode="r")
        cached = self.store.get_metric_array("force_norm")
        expected = np.sqrt(np.square(np.asarray(raw[:5, 0]), dtype=np.float64).sum(axis=(-1, -2)))
        self.assertTrue(np.allclose(cached[:5, 0], expected))

    def test_dipole_magnitude_matches_direct_computation(self) -> None:
        raw = np.load("acn_D.npy", mmap_mode="r")
        cached = self.store.get_metric_array("dipole_magnitude")
        expected = np.sqrt(np.square(np.asarray(raw[:5, 2]), dtype=np.float64).sum(axis=-1))
        self.assertTrue(np.allclose(cached[:5, 2], expected))

    def test_nacr_norm_matches_direct_computation(self) -> None:
        raw = np.load("acn_NACR.npy", mmap_mode="r")
        cached = self.store.get_metric_array("nacr_norm")
        expected = np.sqrt(np.square(np.asarray(raw[:5, 9]), dtype=np.float64).sum(axis=(-1, -2)))
        self.assertTrue(np.allclose(cached[:5, 9], expected))

    def test_denacr_definition_matches_energy_scaled_nacr(self) -> None:
        state_ids = discover_state_ids()
        pair_labels = build_pair_labels(state_ids)
        pair_index = pair_labels.index("1-2")
        denacr_raw = np.load("acn_dENACR.npy", mmap_mode="r")
        nacr_raw = np.load("acn_NACR.npy", mmap_mode="r")
        s1 = np.load("acn_S1.npy", mmap_mode="r")
        s2 = np.load("acn_S2.npy", mmap_mode="r")

        expected = (np.asarray(s2[:5]) - np.asarray(s1[:5]))[:, None] * np.asarray(nacr_raw[:5, pair_index]).reshape(5, -1)
        self.assertTrue(np.allclose(np.asarray(denacr_raw[:5, pair_index]), expected))

    def test_histogram_counts_cover_selected_window(self) -> None:
        histogram = self.store.get_histogram_payload(ids=["S0", "S1"], start=100, end=3100, bins=40)
        for series in histogram["series"]:
            self.assertEqual(sum(series["counts"]), 3000)

    def test_statistics_payload_matches_windowed_mean(self) -> None:
        statistics = self.store.get_statistics_payload(
            start=0,
            end=100,
            state_ids=["S0", "S1"],
            pair_ids=["1-2", "1-3"],
        )
        energy_stats = statistics["metrics"]["energy"]["series"][0]
        raw = np.load("acn_S0.npy", mmap_mode="r")
        self.assertAlmostEqual(energy_stats["mean"], float(np.mean(np.asarray(raw[:100], dtype=np.float64))))

        nacr_stats = statistics["metrics"]["nacr_norm"]["aggregate"]
        self.assertGreater(nacr_stats["count"], 0)

    def test_z_cache_rebuilds_when_energy_source_changes(self) -> None:
        original_root = dashboard_common.ROOT
        original_enabled = tuple(dashboard_common.ENABLED_METRICS)
        original_output = dashboard_common.EXPORT_DIR
        original_overrides = copy.deepcopy(dashboard_common.RUNTIME_CONFIG_OVERRIDES)
        raw_store = None
        shifted_store = None

        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent,
            ignore_cleanup_errors=True,
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            raw = np.arange(20, dtype=np.float64).reshape(5, 4)
            shifted = raw + 125.0
            np.save(temp_dir / "acn_S.npy", raw)
            np.save(temp_dir / "acn_sE.npy", shifted)

            try:
                configure_runtime(data_dir=temp_dir, enabled_metrics=["energy"])
                raw_store = DashboardStore()
                raw_cached = np.asarray(raw_store.get_metric_array("energy")).copy()
                self.assertTrue(np.allclose(raw_cached, raw))
                del raw_store
                raw_store = None
                gc.collect()

                configure_runtime(
                    data_dir=temp_dir,
                    enabled_metrics=["energy"],
                    runtime_overrides={"files": {"energy": {"source": "acn_sE.npy"}}},
                )
                shifted_store = DashboardStore()
                shifted_cached = np.asarray(shifted_store.get_metric_array("energy")).copy()
                self.assertTrue(np.allclose(shifted_cached, shifted))
                self.assertTrue(
                    shifted_store.metadata["cache_inputs"]["metric_inputs"]["energy"]["source"]["path"].endswith("acn_sE.npy")
                )
            finally:
                configure_runtime(
                    data_dir=original_root,
                    enabled_metrics=list(original_enabled),
                    output_dir=original_output,
                    runtime_overrides=original_overrides,
                )
                del raw_store
                del shifted_store
                gc.collect()


if __name__ == "__main__":
    unittest.main()
