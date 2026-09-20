"""Loads and validates config/locations.json.

Product markers and shelf markers are declared as separate marker types (own dictionary, ID range
and physical size) so the two can never be confused, and nothing else in the code base hard-codes
IDs, sizes or positions.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

from aruco.detector import SUPPORTED_DICTIONARIES
from navigation.map import RoomMap
from products.product_manager import ProductManager
from shelves.shelf_manager import ShelfManager

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locations.json")
KINDS = ("product", "shelf")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class MarkerType:
    kind: str
    dictionary: str
    id_min: int
    id_max: int
    size_cm: float

    def owns(self, dictionary: str, marker_id: int) -> bool:
        return dictionary == self.dictionary and self.id_min <= marker_id <= self.id_max


@dataclass
class WarehouseConfig:
    marker_types: Dict[str, MarkerType]
    products: ProductManager
    shelves: ShelfManager
    room_map: RoomMap
    standoff_cm: float
    approach_tolerance_cm: float

    def classify(self, dictionary: str, marker_id: int) -> Optional[MarkerType]:
        for marker_type in self.marker_types.values():
            if marker_type.owns(dictionary, marker_id):
                return marker_type
        return None

    def dictionaries(self):
        return sorted({t.dictionary for t in self.marker_types.values()})


def load_config(path: str = DEFAULT_CONFIG_PATH) -> WarehouseConfig:
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise ConfigError("Cannot read config {}: {}".format(path, e))
    try:
        return _build(raw)
    except KeyError as e:
        raise ConfigError("Config is missing required field {}".format(e))
    except (TypeError, ValueError) as e:
        raise ConfigError("Config has an invalid value: {}".format(e))


def _build(raw: dict) -> WarehouseConfig:
    marker_types = {}
    for kind in KINDS:
        entry = raw["marker_types"][kind]
        low, high = entry["id_range"]
        if entry["dictionary"] not in SUPPORTED_DICTIONARIES:
            raise ConfigError("{}: unknown dictionary '{}' (supported: {})".format(
                kind, entry["dictionary"], sorted(SUPPORTED_DICTIONARIES)))
        if low > high or float(entry["marker_size_cm"]) <= 0:
            raise ConfigError("{}: invalid id_range or marker_size_cm".format(kind))
        marker_types[kind] = MarkerType(kind, entry["dictionary"], int(low), int(high),
                                        float(entry["marker_size_cm"]))

    product_t, shelf_t = marker_types["product"], marker_types["shelf"]
    if (product_t.dictionary == shelf_t.dictionary
            and product_t.id_min <= shelf_t.id_max and shelf_t.id_min <= product_t.id_max):
        raise ConfigError("Product and shelf ID ranges overlap within dictionary {}".format(product_t.dictionary))

    products = ProductManager.from_dict(raw["products"])
    shelves = ShelfManager.from_dict(raw["shelves"])

    for product in products.all():
        if not product_t.owns(product_t.dictionary, product.marker_id):
            raise ConfigError("Product marker {} is outside the product ID range {}-{}".format(
                product.marker_id, product_t.id_min, product_t.id_max))
        if shelves.get(product.shelf_id) is None:
            raise ConfigError("Product {} ({}) targets unknown shelf '{}'".format(
                product.marker_id, product.name, product.shelf_id))
    seen = set()
    for shelf in shelves.all():
        if not shelf_t.owns(shelf_t.dictionary, shelf.marker_id):
            raise ConfigError("Shelf {} marker {} is outside the shelf ID range {}-{}".format(
                shelf.shelf_id, shelf.marker_id, shelf_t.id_min, shelf_t.id_max))
        if shelf.marker_id in seen:
            raise ConfigError("Shelf marker {} is used by more than one shelf".format(shelf.marker_id))
        seen.add(shelf.marker_id)

    room_map = RoomMap.from_dict(raw["map"], shelves)
    if not room_map.is_free(room_map.robot_start):
        raise ConfigError("robot_start {} is not on free space".format(room_map.robot_start))
    for shelf in shelves.all():
        if not room_map.is_free(shelf.approach):
            raise ConfigError("Shelf {} approach point {} is blocked or off the map".format(
                shelf.shelf_id, shelf.approach))

    alignment = raw.get("alignment", {})
    standoff_cm = float(alignment.get("standoff_cm", 30.0))
    # How much further than the stand-off the measured distance may still read and count as arrived.
    # Loosen it if the printed marker size or an uncalibrated camera makes the exact band hard to hit.
    approach_tolerance_cm = float(alignment.get("approach_tolerance_cm", 10.0))
    return WarehouseConfig(marker_types, products, shelves, room_map, standoff_cm, approach_tolerance_cm)
