"""Writes printable marker PNGs for every product and shelf in config/locations.json.

Markers are generated from the dictionary named in the config, so they are guaranteed to match
what the detector looks for. Do not use markers from random online generators unless you have
checked their dictionary (e.g. "4x4", "original") matches the config.

  python tools/generate_markers.py            # writes ./markers/product_101.png, shelf_A_201.png, ...
"""
import os
import sys

import cv2
from cv2 import aruco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aruco.detector import SUPPORTED_DICTIONARIES  # noqa: E402
from config.loader import load_config  # noqa: E402

PIXELS = 600     # marker image side; print at the physical size given in the config
QUIET_ZONE = 100  # white border, required for reliable detection


def write(name, dict_name, marker_id, out_dir, size_cm):
    dictionary = aruco.getPredefinedDictionary(SUPPORTED_DICTIONARIES[dict_name])
    image = aruco.generateImageMarker(dictionary, marker_id, PIXELS)
    image = cv2.copyMakeBorder(image, QUIET_ZONE, QUIET_ZONE, QUIET_ZONE, QUIET_ZONE, cv2.BORDER_CONSTANT, value=255)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, image)
    print("{}  ({} id {}, print the black square {} cm wide)".format(path, dict_name, marker_id, size_cm))


def main():
    config = load_config()
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "markers"
    os.makedirs(out_dir, exist_ok=True)
    product_t, shelf_t = config.marker_types["product"], config.marker_types["shelf"]
    for product in config.products.all():
        write("product_{}.png".format(product.marker_id), product_t.dictionary, product.marker_id, out_dir, product_t.size_cm)
    for shelf in config.shelves.all():
        write("shelf_{}_{}.png".format(shelf.shelf_id, shelf.marker_id), shelf_t.dictionary, shelf.marker_id, out_dir, shelf_t.size_cm)


if __name__ == "__main__":
    main()
