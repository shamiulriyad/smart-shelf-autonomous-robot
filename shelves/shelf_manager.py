"""Shelf database: shelf ID -> fixed marker ID and map position."""
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Shelf:
    shelf_id: str
    marker_id: int
    x: float                       # shelf position on the room map (metres)
    y: float
    approach: Tuple[float, float]  # where the robot should stand to face the shelf (metres)


class ShelfManager:
    def __init__(self, shelves: Iterable[Shelf]):
        self._by_id: Dict[str, Shelf] = {s.shelf_id: s for s in shelves}
        self._by_marker: Dict[int, Shelf] = {s.marker_id: s for s in self._by_id.values()}

    @classmethod
    def from_dict(cls, data: dict) -> "ShelfManager":
        return cls(
            Shelf(str(shelf_id), int(e["marker_id"]), float(e["x"]), float(e["y"]),
                  (float(e["approach"]["x"]), float(e["approach"]["y"])))
            for shelf_id, e in data.items()
        )

    def get(self, shelf_id: str) -> Optional[Shelf]:
        return self._by_id.get(shelf_id)

    def get_by_marker(self, marker_id: int) -> Optional[Shelf]:
        return self._by_marker.get(marker_id)

    def all(self):
        return sorted(self._by_id.values(), key=lambda s: s.shelf_id)
