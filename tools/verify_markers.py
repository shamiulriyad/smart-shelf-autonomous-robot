"""Debug command: check that generated markers are actually detected by the project's detector.

  python tools/verify_markers.py                                   # every marker PNG in ./markers, one by one
  python tools/verify_markers.py --dir some/folder                 # PNGs from another folder
  python tools/verify_markers.py --pdf markers/pdf/all_markers.pdf # every marker found on every PDF page

The PNG check is driven by config/locations.json: each configured product/shelf must have its PNG
(named the way tools/generate_markers.py names it), and detecting that PNG must yield exactly that
marker ID in the configured dictionary. Exit code is 0 only if everything passes.

The detect/rasterise helpers here are also used by tools/generate_marker_pdf.py.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aruco.detector import MarkerDetector  # noqa: E402
from config.loader import DEFAULT_CONFIG_PATH, ConfigError, load_config  # noqa: E402

DEFAULT_PNG_DIR = os.path.join(ROOT, "markers")
_detectors = {}


def detect_markers(image, dictionary):
    """Runs the project's own MarkerDetector, restricted to one dictionary."""
    if dictionary not in _detectors:
        _detectors[dictionary] = MarkerDetector(dictionaries=[dictionary])
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return [d for d in _detectors[dictionary].detect(image) if d.dictionary == dictionary]


def check_single_marker(image, dictionary, marker_id):
    """Returns None if `image` contains exactly the expected marker, else a description of the problem."""
    found = detect_markers(image, dictionary)
    ids = sorted(d.marker_id for d in found)
    if ids == [marker_id]:
        return None
    if not ids:
        return "no {} marker detected".format(dictionary)
    return "expected id {} but detected {}".format(marker_id, ids)


def rasterize_pdf(path, dpi):
    """Yields (page_index, BGR image, (width_pt, height_pt)) for each page of a PDF."""
    import pymupdf  # PyMuPDF: renders the PDF exactly as a viewer would
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            yield index, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (page.rect.width, page.rect.height)


def expected_pngs(config):
    """(filename, dictionary, marker_id, description) for every configured marker, using generate_markers.py naming."""
    product_t, shelf_t = config.marker_types["product"], config.marker_types["shelf"]
    for product in config.products.all():
        yield ("product_{}.png".format(product.marker_id), product_t.dictionary, product.marker_id,
               "product {}".format(product.name))
    for shelf in config.shelves.all():
        yield ("shelf_{}_{}.png".format(shelf.shelf_id, shelf.marker_id), shelf_t.dictionary, shelf.marker_id,
               "shelf {}".format(shelf.shelf_id))


def verify_pngs(config, directory):
    failures = 0
    expected_names = set()
    for name, dictionary, marker_id, what in expected_pngs(config):
        expected_names.add(name)
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            problem = "file missing (run: python tools/generate_markers.py)"
        else:
            image = cv2.imread(path)
            problem = "cannot read image" if image is None else check_single_marker(image, dictionary, marker_id)
        if problem:
            failures += 1
        print("{}  {}  ({}, {} id {})".format("FAIL" if problem else "PASS", path, what, dictionary, marker_id)
              + ("  -> " + problem if problem else ""))
    for path in sorted(glob.glob(os.path.join(directory, "*.png"))):
        if os.path.basename(path) not in expected_names:
            print("NOTE  {}  is not in config/locations.json (not checked)".format(path))
    return failures


def verify_pdf(config, path, dpi):
    failures = 0
    total = 0
    for index, image, _ in rasterize_pdf(path, dpi):
        for dictionary in config.dictionaries():
            for det in sorted(detect_markers(image, dictionary), key=lambda d: (d.center[1], d.center[0])):
                total += 1
                marker_type = config.classify(dictionary, det.marker_id)
                known = (config.products.get_by_marker(det.marker_id) if marker_type and marker_type.kind == "product"
                         else config.shelves.get_by_marker(det.marker_id) if marker_type else None)
                measured_cm = det.side_px * 2.54 / dpi
                if known is None:
                    failures += 1
                    status = "FAIL  not a configured product/shelf marker"
                else:
                    off = abs(measured_cm - marker_type.size_cm) / marker_type.size_cm > 0.05
                    status = "PASS  {}".format(marker_type.kind) + (
                        "  (size differs from config's {} cm - custom size?)".format(marker_type.size_cm) if off else "")
                print("page {}  id {:<4} {:5.2f} cm  {}".format(index + 1, det.marker_id, measured_cm, status))
    if total == 0:
        failures += 1
        print("FAIL  no markers detected in {}".format(path))
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=DEFAULT_PNG_DIR, help="folder with the marker PNGs (default: ./markers)")
    parser.add_argument("--pdf", help="verify this PDF instead of the PNGs")
    parser.add_argument("--dpi", type=int, default=200, help="rasterisation DPI for --pdf (default 200)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print("error: {}".format(e))
        return 1
    failures = verify_pdf(config, args.pdf, args.dpi) if args.pdf else verify_pngs(config, args.dir)
    print("\n{}".format("All markers detected." if not failures else "{} problem(s) found.".format(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
