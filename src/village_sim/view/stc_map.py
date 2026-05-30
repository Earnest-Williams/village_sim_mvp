"""StyledTextCtrl map content helpers shared by wx GUI tests."""

from __future__ import annotations

from village_sim.view.ascii_view import RenderedMap

GUI_DEFAULT_WORLD_SIZE: int = 256
MAP_BACKGROUND: str = "#5a422d"
MAP_DEFAULT_FOREGROUND: str = "#e8dcc4"
MAP_STATUS_FOREGROUND: str = "#eadfcb"
MAP_DEFAULT_FONT_POINT_SIZE: int = 9

# Maps each semantic glyph role to an STC style number.
# Style 0 is left as the STC default (used for newlines and unknown roles).
STC_ROLE_STYLE: dict[str, int] = {
    "summary": 1,
    "agent": 2,
    "agent_sleeping": 3,
    "water": 4,
    "broadleaf": 5,
    "evergreen": 6,
    "grass": 7,
    "brush": 8,
    "wetland": 9,
    "food": 10,
    "rock": 11,
    "cave": 12,
    "hill": 13,
}


def build_stc_content(
    rendered_map: RenderedMap, *, left_padding_columns: int = 0
) -> tuple[str, list[tuple[int, int]]]:
    """Return ``(full_text, style_runs)`` for STC rendering.

    *style_runs* is a list of ``(utf8_byte_count, style_number)`` pairs
    describing consecutive same-style segments. Multi-byte Unicode characters
    are accounted for correctly: every byte in a glyph receives one style.
    """
    text_parts: list[str] = []
    style_runs: list[tuple[int, int]] = []
    current_style: int = -1
    current_bytes: int = 0
    padding: str = " " * max(0, left_padding_columns)

    def append_text(text: str, style: int) -> None:
        nonlocal current_style, current_bytes
        if text == "":
            return
        text_parts.append(text)
        byte_len = len(text.encode("utf-8"))
        if style == current_style:
            current_bytes += byte_len
            return
        if current_bytes > 0:
            style_runs.append((current_bytes, current_style))
        current_style = style
        current_bytes = byte_len

    summary_style = STC_ROLE_STYLE["summary"]
    for line in _wx_header_lines(rendered_map):
        append_text(padding, 0)
        append_text(line, summary_style)
        append_text("\n", 0)
    for row in rendered_map.rows:
        append_text(padding, 0)
        for glyph in row:
            append_text(glyph.char, STC_ROLE_STYLE.get(glyph.role, 0))
        append_text("\n", 0)
    if current_bytes > 0:
        style_runs.append((current_bytes, current_style))
    return "".join(text_parts), style_runs


def _wx_header_lines(rendered_map: RenderedMap) -> list[str]:
    """Split wx-only headers so they do not dominate horizontal scroll width."""
    status_parts = rendered_map.status.split(" health=", maxsplit=1)
    lines: list[str] = [status_parts[0]]
    if len(status_parts) == 2:
        lines.append(f"Needs: health={status_parts[1]}")

    legend_parts = rendered_map.legend.split(" | Scale: ", maxsplit=1)
    legend = legend_parts[0]
    legend_items = (
        legend.removeprefix("Legend: ").split(", ")
        if legend.startswith("Legend: ")
        else legend.split(", ")
    )
    if len(legend_items) > 5:
        lines.append("Legend: " + ", ".join(legend_items[:5]))
        lines.append("        " + ", ".join(legend_items[5:]))
    else:
        lines.append(legend)
    if len(legend_parts) == 2:
        lines.append(f"Scale: {legend_parts[1]}")
    return lines
