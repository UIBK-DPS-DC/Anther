import tempfile
import unittest
from pathlib import Path

import numpy as np

from hardware_data import read_power_trace, read_transfer_log, timestamp_step


class HardwareDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir="/tmp/opencode" if Path("/tmp/opencode").is_dir() else None,
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def test_integrates_raw_samples_and_discards_stale_prefix(self):
        rows = ["timestamp (us),current (mA)", "1000000000,80", "broken,row", "1,2,3"]
        rows += [f"{i * 1000},100" for i in range(101)]
        trace = read_power_trace(self.write("power.csv", "\n".join(rows)))
        self.assertAlmostEqual(trace.time_s[-1], 0.1)
        self.assertAlmostEqual(trace.energy_j[-1], 0.05)
        self.assertEqual(trace.rejected_rows, 3)
        self.assertEqual(trace.samples, 101)

    def test_window_clipping_interpolates_boundaries(self):
        path = self.write("crop.csv", "time (s),current (mA)\n0,0\n1,200\n2,0\n")
        trace = read_power_trace(path, window={"start_s": 0.5, "end_s": 1.5})
        np.testing.assert_allclose(trace.time_s[[0, -1]], [0, 1])
        self.assertAlmostEqual(trace.energy_j[-1], 0.75)

    def test_drops_isolated_corruption_without_losing_subsequent_samples(self):
        rows = ["timestamp (us),current (mA)"]
        rows += [f"{i * 1000},100" for i in range(50)]
        rows += ["3000000000,100", "50000,999", "49000,100", "not,a,sample"]
        rows += [f"{i * 1000},100" for i in range(50, 101)]
        trace = read_power_trace(self.write("power.csv", "\n".join(rows)))
        self.assertAlmostEqual(trace.energy_j[-1], 0.05)
        self.assertEqual(trace.rejected_rows, 4)

    def test_requires_coverage_for_explicit_window(self):
        path = self.write("crop.csv", "time (s),current (mA)\n0,100\n1,100\n")
        with self.assertRaisesRegex(ValueError, "does not cover"):
            read_power_trace(path, window={"start_s": 0, "end_s": 2})

    def test_reads_logged_duration_not_model_predictions(self):
        path = self.write("run.log", "\n".join([
            "[FILE] 51200B  CRC32=0x1234  lambda=0.650  confirmed=False",
            "Latency : 100 s",
            "[CTRL] Switched to Class A",
            "[CTRL] Switched to Class C",
            "[FILE] Done: 917 chunks, 6872.71s, class=C, CRC32=0x1234",
        ]))
        result = read_transfer_log(path)
        self.assertEqual(result["payload_bytes"], 51200)
        self.assertEqual(result["transfer_duration_s"], 6872.71)
        self.assertEqual(result["class_sequence"], ["A", "C"])

    def test_incomplete_log_is_rejected(self):
        path = self.write("run.log", "[FILE] 51200B CRC32=0x1234 lambda=0.650")
        with self.assertRaisesRegex(ValueError, "completed transfer"):
            read_transfer_log(path)

    def test_only_plausible_timestamp_wraps_are_unwrapped(self):
        self.assertEqual(timestamp_step(2**32 - 500, 500), (1000, 1))
        self.assertIsNone(timestamp_step(2000, 1000))
        self.assertIsNone(timestamp_step(1000, 3_000_000_000))
        self.assertEqual(timestamp_step(2.0, 2.5, unwrapped=True), (0.5, 0))


if __name__ == "__main__":
    unittest.main()
