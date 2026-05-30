"""CLI help text tests."""

from __future__ import annotations

import unittest

from village_sim.run import build_parser


class TestCliHelp(unittest.TestCase):
    def test_discoverables_help_lists_all_seeded_ids(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("spring_001", help_text)
        self.assertIn("berry_bush_001", help_text)
        self.assertIn("cave_001", help_text)

    def test_agents_help_is_listed(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("--agents", help_text)

    def test_agents_argument_is_accepted(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--agents", "40"])

        self.assertEqual(args.agents, 40)


if __name__ == "__main__":
    unittest.main()
