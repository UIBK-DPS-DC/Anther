import unittest
from unittest.mock import patch

from matplotlib.text import Text

from plot_style import FONT_SIZE, IEEE_COLUMN_WIDTH, IEEE_TEXT_WIDTH, LEGEND_GAP_PT, plt, save_figure


class PaperLayoutTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_final_width_and_font_size_for_each_panel_count(self):
        for count in (1, 2, 4):
            with self.subTest(panels=count):
                fig, axes = plt.subplots(1, count, squeeze=False, figsize=(16, 4))
                for ax in axes.flat:
                    ax.plot([1, 2, 3], [1, 2, 4], label="Class A")
                    ax.set_xlabel("File Size (KiB)", fontsize=11)
                    ax.set_ylabel("Energy (J)", fontsize=11)
                    ax.set_title("Measured energy")
                fig.legend(*axes.flat[0].get_legend_handles_labels(), ncol=3)
                with patch.object(fig, "savefig") as export:
                    save_figure(fig, "unused.pdf")
                box = export.call_args.kwargs["bbox_inches"]
                self.assertAlmostEqual(box.width, IEEE_TEXT_WIDTH if count == 4 else IEEE_COLUMN_WIDTH)
                if count <= 2:
                    self.assertLessEqual(box.height, box.width)
                self.assertTrue(all(t.get_fontsize() == FONT_SIZE for t in fig.findobj(Text)))
                renderer = fig.canvas.get_renderer()
                for ax in axes.flat:
                    bounds = ax.get_tightbbox(renderer)
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertLessEqual(bounds.x1, fig.bbox.width)

    def test_legends_fit_and_stack_below_labels(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2], label="Total Time Overhead (w/ Control)")
        ax.set_xlabel("File Size (KiB)")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles * 3, labels * 3, ncol=3, title="Update Method")
        fig.legend(handles, [r"Anther ($\lambda=0.65$)"], title="Architecture")
        with patch.object(fig, "savefig"):
            save_figure(fig, "unused.pdf")
        renderer = fig.canvas.get_renderer()
        previous_bottom = ax.get_tightbbox(renderer).y0
        for legend in fig.legends:
            bounds = legend.get_window_extent(renderer)
            self.assertGreaterEqual(bounds.x0, 0)
            self.assertLessEqual(bounds.x1, fig.bbox.width)
            self.assertAlmostEqual((previous_bottom - bounds.y1) * 72 / fig.dpi, LEGEND_GAP_PT)
            previous_bottom = bounds.y0

    def test_boxed_notes_do_not_cover_data_at_column_width(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="Measured energy")
        ax.text(0.03, 0.97, "Full recording, including setup/reset", transform=ax.transAxes,
                va="top", bbox={"facecolor": "white"})
        fig.legend(*ax.get_legend_handles_labels())
        with patch.object(fig, "savefig") as export:
            save_figure(fig, "unused.pdf")
        self.assertEqual(len(ax.texts), 0)
        self.assertEqual(len(fig.texts), 1)
        renderer = fig.canvas.get_renderer()
        note = fig.texts[0].get_window_extent(renderer)
        self.assertLess(note.y1, ax.get_tightbbox(renderer).y0)
        self.assertGreater(note.y0, fig.legends[0].get_window_extent(renderer).y1)
        self.assertGreaterEqual(note.x0, 0)
        self.assertLessEqual(note.x1, fig.bbox.width)
        self.assertLess(export.call_args.kwargs["bbox_inches"].y0 * fig.dpi, note.y0)

    def test_half_column_trajectories_preserve_print_font_size(self):
        fig, ax = plt.subplots()
        ax.plot([0, 100, 200], [0, 1, 2], label="Anther")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Energy (kJ)")
        ax.set_yticks([0, 1, 2])
        ax.legend(loc="upper right")
        ax.text(0.03, 0.97, "Setup/reset included", transform=ax.transAxes,
                va="top", bbox={"facecolor": "white"})
        with patch.object(fig, "savefig") as export:
            save_figure(fig, "unused.pdf", column_fraction=0.475)
        box = export.call_args.kwargs["bbox_inches"]
        self.assertAlmostEqual(box.width, 0.475 * IEEE_COLUMN_WIDTH)
        self.assertLess(box.height, 1.5)
        self.assertTrue(all(t.get_fontsize() == FONT_SIZE for t in fig.findobj(Text)))

    def test_long_vertical_labels_wrap_in_short_plots(self):
        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.plot([1, 2], [1, 2])
            ax.set_ylabel("Relative completion time overhead versus Fixed Class C (%)")
            ax.set_xlabel("File Size (KiB)")
        with patch.object(fig, "savefig") as export:
            save_figure(fig, "unused.pdf")
        renderer = fig.canvas.get_renderer()
        bounds = export.call_args.kwargs["bbox_inches"]
        for ax in axes:
            self.assertIn("\n", ax.get_ylabel())
            label = ax.yaxis.label.get_window_extent(renderer)
            self.assertGreaterEqual(label.y0 / fig.dpi, bounds.y0)
            self.assertLessEqual(label.y1 / fig.dpi, bounds.y1)

    def test_legend_markers_are_small_with_light_outlines(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], marker="D", markersize=7, markeredgewidth=1.5, label="Reference")
        ax.legend()
        fig.legend(*ax.get_legend_handles_labels())
        with patch.object(fig, "savefig"):
            save_figure(fig, "unused.pdf")
        for legend in [ax.get_legend(), *fig.legends]:
            for handle in legend.legend_handles:
                self.assertLessEqual(handle.get_markersize(), 3.5)
                self.assertLessEqual(handle.get_markeredgewidth(), 0.5)


if __name__ == "__main__":
    unittest.main()
