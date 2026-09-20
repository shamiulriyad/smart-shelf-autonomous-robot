import os
import sys
import tempfile
import unittest

import numpy as np

import tests.conftest  # noqa: F401  (puts the repo root on sys.path)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import generate_marker_pdf as gen  # noqa: E402
import verify_markers  # noqa: E402
from config.loader import load_config  # noqa: E402


class MarkerPdfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_gen(self, *args):
        out = os.path.join(self.tmp.name, "out.pdf")
        return gen.main(list(args) + ["--output", out]), out

    def test_all_markers_are_generated_and_verified(self):
        code, out = self.run_gen("--all")
        self.assertEqual(code, 0)
        cfg = load_config()
        product_layout = gen.plan_layout(cfg.marker_types["product"].size_cm * 10 * gen.MM, "product")
        product_pages = -(-len(cfg.products.all()) // (product_layout.cols * product_layout.rows))
        self.assertEqual(len(list(verify_markers.rasterize_pdf(out, 72))), product_pages + len(cfg.shelves.all()))
        self.assertEqual(verify_markers.main(["--pdf", out]), 0)

    def test_selected_ids_only(self):
        code, out = self.run_gen("--product-ids", "102")
        self.assertEqual(code, 0)
        self.assertEqual(len(list(verify_markers.rasterize_pdf(out, 72))), 1)

    def test_unconfigured_or_wrong_type_ids_are_rejected_and_nothing_is_written(self):
        for args in (["--product-ids", "199"], ["--product-ids", "203"], ["--shelf-ids", "299"]):
            code, out = self.run_gen(*args)
            self.assertEqual(code, 1, args)
            self.assertFalse(os.path.exists(out), args)
            self.assertFalse(os.path.exists(out + ".tmp"), args)

    def test_marker_too_large_for_a4_is_rejected(self):
        code, _ = self.run_gen("--shelves", "--shelf-size-cm", "25")
        self.assertEqual(code, 1)

    def test_layout_follows_marker_size(self):
        product = gen.plan_layout(5 * 10 * gen.MM, "product")
        shelf = gen.plan_layout(18.7 * 10 * gen.MM, "shelf")
        self.assertEqual(product.cols, 3)
        self.assertEqual((shelf.cols, shelf.rows), (1, 1))
        self.assertEqual(gen.plan_layout(3 * 10 * gen.MM, "product").cols, 4)

    def test_marker_uses_the_configured_dictionary(self):
        cfg = load_config()
        dictionary = cfg.marker_types["product"].dictionary
        self.assertEqual(dictionary, "ARUCO_ORIGINAL")
        image = np.kron(gen.marker_grid(dictionary, 102), np.ones((20, 20), np.uint8))
        image = np.pad(image, 20, constant_values=255)
        self.assertIsNone(verify_markers.check_single_marker(image, dictionary, 102))
        self.assertIsNotNone(verify_markers.check_single_marker(image, dictionary, 101))

    def test_verifier_rejects_a_blank_image(self):
        blank = np.full((300, 300), 255, np.uint8)
        self.assertIsNotNone(verify_markers.check_single_marker(blank, "ARUCO_ORIGINAL", 102))


if __name__ == "__main__":
    unittest.main()
