"""Error-analysis helpers; no model or checkpoint needed."""
import unittest
from wlm.compare import align, describe, mcnemar, status
from wlm.data import tokens
from wlm.metrics import edit_distance


class CompareTests(unittest.TestCase):
    def test_alignment_matches_edit_distance(self):
        for truth, prediction in [("x ^ { 2 } + 1", "x ^ { 2 } + l"), ("\\frac { a } { b }", "a / b"),
                                  ("a", "a + b"), ("", "x"), ("x + y", "")]:
            ops = align(tokens(truth), tokens(prediction))
            errors = sum(op != "=" for op, _, _ in ops)
            self.assertEqual(errors, edit_distance(tokens(truth), tokens(prediction)))

    def test_error_types(self):
        _, types, confusions = describe("x ^ { 2 } + 1", "x ^ { 2 } + l")
        self.assertEqual(types, {"symbol_substitution": 1})
        self.assertEqual(confusions, {"1 -> l": 1})
        _, types, _ = describe("x ^ { 2 }", "x 2")
        self.assertEqual(set(types), {"layout", "braces"})
        _, types, _ = describe("a + b", "a +")
        self.assertEqual(types, {"missing_symbol": 1})

    def test_mcnemar_and_status(self):
        self.assertEqual(mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar(8, 0), 2 / 2 ** 8)
        self.assertEqual(mcnemar(3, 3), 1.0)
        self.assertEqual(status(2, 0), "improved")
        self.assertEqual(status(0, 1), "regressed")
        self.assertEqual(status(0, None), "correct")


if __name__ == "__main__":
    unittest.main()
