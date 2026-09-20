"""Room map: a 2D occupancy grid built from the config (shelves + obstacles).

Coordinates are metres in the map frame. This layer knows nothing about cameras or robots.
"""
from dataclasses import dataclass
from math import floor
from typing import Iterable, List, Optional, Tuple

import numpy as np

from shelves.shelf_manager import ShelfManager

XY = Tuple[float, float]


@dataclass(frozen=True)
class Rect:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def inflated(self, margin: float) -> "Rect":
        return Rect(self.x_min - margin, self.y_min - margin, self.x_max + margin, self.y_max + margin)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class RoomMap:
    def __init__(self, width_m: float, height_m: float, cell_size_m: float,
                 robot_start: XY, blocked_rects: Iterable[Rect], robot_radius_m: float = 0.0,
                 shelf_depth_m: float = 0.5):
        self.width_m = width_m
        self.height_m = height_m
        self.cell_size_m = cell_size_m
        self.robot_start = robot_start
        self.shelf_depth_m = shelf_depth_m
        self.cols = int(round(width_m / cell_size_m))
        self.rows = int(round(height_m / cell_size_m))

        # Obstacles are inflated by the robot radius so the planner can treat the robot as a point.
        self.blocked = np.zeros((self.rows, self.cols), dtype=bool)
        inflated = [r.inflated(robot_radius_m) for r in blocked_rects]
        for row in range(self.rows):
            for col in range(self.cols):
                cx, cy = self.cell_to_world((col, row))
                if any(r.contains(cx, cy) for r in inflated):
                    self.blocked[row, col] = True

    @classmethod
    def from_dict(cls, data: dict, shelves: ShelfManager) -> "RoomMap":
        rects: List[Rect] = [Rect(o["x_min"], o["y_min"], o["x_max"], o["y_max"])
                             for o in data.get("obstacles", [])]
        footprint = data.get("shelf_footprint_m", {"width": 1.0, "depth": 0.5})
        half_w, half_d = footprint["width"] / 2, footprint["depth"] / 2
        for shelf in shelves.all():
            rects.append(Rect(shelf.x - half_w, shelf.y - half_d, shelf.x + half_w, shelf.y + half_d))
        start = data["robot_start"]
        return cls(data["width_m"], data["height_m"], data["cell_size_m"],
                   (start["x"], start["y"]), rects, data.get("robot_radius_m", 0.0), footprint["depth"])

    def world_to_cell(self, xy: XY) -> Tuple[int, int]:
        return floor(xy[0] / self.cell_size_m), floor(xy[1] / self.cell_size_m)

    def cell_to_world(self, cell: Tuple[int, int]) -> XY:
        return (cell[0] + 0.5) * self.cell_size_m, (cell[1] + 0.5) * self.cell_size_m

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.cols and 0 <= cell[1] < self.rows

    def is_free_cell(self, cell: Tuple[int, int]) -> bool:
        return self.in_bounds(cell) and not self.blocked[cell[1], cell[0]]

    def is_free(self, xy: XY) -> bool:
        return self.is_free_cell(self.world_to_cell(xy))

    def render_ascii(self, path: Optional[List[XY]] = None, shelves: Optional[ShelfManager] = None) -> str:
        """Text picture of the map (y grows downward, like image rows). '#' blocked, '.' path."""
        canvas = [["#" if self.blocked[r, c] else " " for c in range(self.cols)] for r in range(self.rows)]
        for x, y in path or []:
            col, row = self.world_to_cell((x, y))
            if self.in_bounds((col, row)):
                canvas[row][col] = "."
        if shelves:
            for shelf in shelves.all():
                col, row = self.world_to_cell((shelf.x, shelf.y))
                if self.in_bounds((col, row)):
                    canvas[row][col] = shelf.shelf_id[0]
        col, row = self.world_to_cell(self.robot_start)
        canvas[row][col] = "R"
        border = "+" + "-" * self.cols + "+"
        return "\n".join([border] + ["|" + "".join(r) + "|" for r in canvas] + [border])
