from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from pathlib import Path

from dashboard_common import build_runtime_overrides_from_input, parse_input_file, resolve_runtime_settings


class DashboardCommonTests(unittest.TestCase):
    def _write_workspace_input(self, name: str, contents: str) -> Path:
        path = Path(__file__).resolve().parent / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_parse_input_file_ignores_comment_lines(self) -> None:
        input_path = self._write_workspace_input(
            "_test_dashboard_comments.inp",
            "# comment\n"
            "data_dir = C:\\data\\trajectory\n"
            "metrics = energies, forces_norm\n"
            "save_formats = png\n",
        )
        parsed = parse_input_file(input_path)

        self.assertEqual(parsed["data_dir"], "C:\\data\\trajectory")
        self.assertEqual(parsed["metrics"], "energies, forces_norm")
        self.assertEqual(parsed["save_formats"], "png")

    def test_build_runtime_overrides_maps_files_labels_and_formats(self) -> None:
        overrides = build_runtime_overrides_from_input(
            {
                "energy_file": "custom_energy.npy",
                "force_prefix": "custom_F",
                "nacr_axis_label": "Custom NACR",
                "save_formats": "pdf",
            }
        )

        self.assertEqual(overrides["files"]["energy"]["source"], "custom_energy.npy")
        self.assertEqual(overrides["files"]["force_norm"]["fallback_prefix"], "custom_F")
        self.assertEqual(overrides["axis_labels"]["nacr_norm"], "Custom NACR")
        self.assertEqual(overrides["export"]["formats"], ["pdf"])

    def test_cli_metric_flags_override_input_file_metrics_with_warning(self) -> None:
        input_path = self._write_workspace_input(
            "_test_dashboard_priority.inp",
            "metrics = all\n"
            "port = 9000\n",
        )

        args = argparse.Namespace(
            input_file=str(input_path),
            data_dir=None,
            host=None,
            port=None,
            all=False,
            energies=True,
            forces_norm=False,
            dipole_magnitude=False,
            nacrs_norm=False,
            scaled_nacrs=False,
        )

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            settings = resolve_runtime_settings(args)

        self.assertEqual(settings["enabled_metrics"], ("energy",))
        self.assertEqual(settings["port"], 9000)
        self.assertIn("CLI metric selection takes priority", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
