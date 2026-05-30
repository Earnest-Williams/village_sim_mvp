from __future__ import annotations

import unittest

from village_sim.view.ascii_view import MapGlyph, RenderedMap, rendered_map_to_text
from village_sim.view.stc_map import (
    GUI_DEFAULT_WORLD_SIZE,
    STC_ROLE_STYLE,
    build_stc_content,
)


class StcMapContentTests(unittest.TestCase):
    def test_unicode_glyph_style_runs_count_utf8_bytes(self) -> None:
        rendered = RenderedMap(
            status="Agent: x=1 y=2 goal=idle action=idle health=1.00 thirst=0.00 hunger=0.00 fatigue=0.00",
            legend="Legend: ♣ broadleaf, ♠ evergreen | Scale: 1 tile ≈ 2m x 2m",
            rows=[[MapGlyph("♣", "broadleaf"), MapGlyph("♠", "evergreen")]],
        )

        full_text, style_runs = build_stc_content(rendered)

        self.assertIn("♣♠", full_text)
        self.assertEqual(
            len(full_text.encode("utf-8")),
            sum(byte_count for byte_count, _ in style_runs),
        )
        broadleaf_runs = [
            byte_count
            for byte_count, style in style_runs
            if style == STC_ROLE_STYLE["broadleaf"]
        ]
        evergreen_runs = [
            byte_count
            for byte_count, style in style_runs
            if style == STC_ROLE_STYLE["evergreen"]
        ]
        self.assertEqual(broadleaf_runs, [len("♣".encode("utf-8"))])
        self.assertEqual(evergreen_runs, [len("♠".encode("utf-8"))])

    def test_left_padding_applies_to_headers_and_rows(self) -> None:
        rendered = RenderedMap(
            status="Agent: x=0 y=0 goal=idle action=idle health=1.00 thirst=0.00 hunger=0.00 fatigue=0.00",
            legend="Legend: @ agent, z sleeping | Scale: 1 tile ≈ 2m x 2m",
            rows=[[MapGlyph("@", "agent")]],
        )

        full_text, style_runs = build_stc_content(rendered, left_padding_columns=3)
        lines = full_text.splitlines()

        self.assertTrue(lines)
        self.assertTrue(all(line.startswith("   ") for line in lines))
        self.assertEqual(
            len(full_text.encode("utf-8")),
            sum(byte_count for byte_count, _ in style_runs),
        )

    def test_wx_header_split_does_not_change_plain_text_renderer(self) -> None:
        rendered = RenderedMap(
            status="Agent: x=0 y=0 goal=idle action=idle health=1.00 thirst=0.00 hunger=0.00 fatigue=0.00",
            legend="Legend: @ agent, z sleeping, ~ stream/water, * food/berries, C cave, ♣ broadleaf, ♠ evergreen | Scale: 1 tile ≈ 2m x 2m",
            rows=[[MapGlyph("@", "agent")]],
        )

        wx_text, _style_runs = build_stc_content(rendered)
        plain_text = rendered_map_to_text(rendered)

        self.assertGreater(len(wx_text.splitlines()), len(plain_text.splitlines()))
        self.assertEqual(plain_text.splitlines()[0], rendered.status)
        self.assertEqual(plain_text.splitlines()[1], rendered.legend)

    def test_wx_header_split_does_not_duplicate_scale_with_short_legend(self) -> None:
        rendered = RenderedMap(
            status="Agent: x=0 y=0 goal=idle action=idle health=1.00 thirst=0.00 hunger=0.00 fatigue=0.00",
            legend="Legend: @ agent, z sleeping | Scale: 1 tile ≈ 2m x 2m",
            rows=[[MapGlyph("@", "agent")]],
        )

        wx_text, _style_runs = build_stc_content(rendered)

        self.assertEqual(wx_text.count("Scale: 1 tile ≈ 2m x 2m"), 1)

    def test_gui_default_world_size_is_256(self) -> None:
        self.assertEqual(GUI_DEFAULT_WORLD_SIZE, 256)


if __name__ == "__main__":
    unittest.main()
