"""Grid A* path planner over a RoomMap."""
import heapq
import math
from typing import Dict, List, Optional, Tuple

from navigation.map import RoomMap, XY

Cell = Tuple[int, int]

_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class PathPlanner:
    def __init__(self, room_map: RoomMap):
        self.map = room_map

    def plan(self, start: XY, goal: XY) -> Optional[List[XY]]:
        """Waypoints (metres) from start to goal, or None if either end is blocked or unreachable."""
        start_cell, goal_cell = self.map.world_to_cell(start), self.map.world_to_cell(goal)
        if not (self.map.is_free_cell(start_cell) and self.map.is_free_cell(goal_cell)):
            return None

        open_heap = [(0.0, start_cell)]
        came_from: Dict[Cell, Cell] = {}
        cost = {start_cell: 0.0}
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_cell:
                return self._to_waypoints(self._reconstruct(came_from, current), start, goal)
            for dx, dy in _MOVES:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.map.is_free_cell(nxt):
                    continue
                # No corner cutting: a diagonal step needs both adjacent cells free.
                if dx and dy and not (self.map.is_free_cell((current[0] + dx, current[1]))
                                      and self.map.is_free_cell((current[0], current[1] + dy))):
                    continue
                new_cost = cost[current] + math.hypot(dx, dy)
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap, (new_cost + self._heuristic(nxt, goal_cell), nxt))
        return None

    @staticmethod
    def _heuristic(a: Cell, b: Cell) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)  # octile distance

    @staticmethod
    def _reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _to_waypoints(self, cells: List[Cell], start: XY, goal: XY) -> List[XY]:
        # Keep only cells where the heading changes; use exact start/goal at the ends.
        keep = [cells[0]]
        for prev, cur, nxt in zip(cells, cells[1:], cells[2:]):
            if (cur[0] - prev[0], cur[1] - prev[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
                keep.append(cur)
        keep.append(cells[-1])
        points = [self.map.cell_to_world(c) for c in keep]
        points[0], points[-1] = start, goal
        return points
