import unittest

from font_support import (
    build_pdf_cjk_font_preamble,
    contains_japanese_script,
)


class PdfCjkFontSupportTests(unittest.TestCase):
    def test_detects_japanese_kana_and_long_vowel_mark(self):
        self.assertTrue(contains_japanese_script("関数を定義する"))
        self.assertTrue(contains_japanese_script("カルーシュ"))
        self.assertFalse(contains_japanese_script("定义一个正定矩阵"))

    def test_japanese_text_prioritizes_japanese_fonts(self):
        preamble = build_pdf_cjk_font_preamble("関数を定義する")
        main_font_switch = preamble[3]

        self.assertLess(
            main_font_switch.index("Noto Serif CJK JP"),
            main_font_switch.index("Noto Serif CJK SC"),
        )
        self.assertIn(r"\setCJKmainfont", main_font_switch)
        self.assertIn(
            r"\setCJKfallbackfamilyfont{\CJKrmdefault}",
            preamble[4],
        )
        self.assertIn("Noto Serif CJK SC", preamble[4])
        self.assertNotIn("Noto Serif CJK JP", preamble[4])
        self.assertIn(r"\xeCJKsetup{AutoFallBack=true}", preamble)

    def test_chinese_text_prioritizes_simplified_chinese_fonts(self):
        preamble = build_pdf_cjk_font_preamble("定义一个正定矩阵")
        main_font_switch = preamble[3]

        self.assertLess(
            main_font_switch.index("Noto Serif CJK SC"),
            main_font_switch.index("Noto Serif CJK JP"),
        )

    def test_user_text_is_not_embedded_in_font_commands(self):
        marker = r"\end{document}"
        preamble = "\n".join(build_pdf_cjk_font_preamble(marker))

        self.assertNotIn(marker, preamble)
        self.assertGreaterEqual(preamble.count(r"\IfFontExistsTF"), 2)


if __name__ == "__main__":
    unittest.main()
