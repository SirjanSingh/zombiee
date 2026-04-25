"""Fixed 15x15 grid layout for SurviveCity v2.

Cell legend (used in the rendered grid string and in clients/prompts):
    '.' empty
    '#' wall
    'F' food depot
    'W' water depot
    'M' medicine depot
    'S' safehouse cell
    'A<id>' agent (rendered on top of the base grid each step)
    'Z' zombie (rendered on top of the base grid each step)
"""

from __future__ import annotations

import copy

GRID_ROWS = 15
GRID_COLS = 15

# Safehouse — 3x3 block in the centre
SAFEHOUSE_CELLS: set[tuple[int, int]] = {
    (r, c) for r in range(6, 9) for c in range(6, 9)
}

# Food depots — 8 cells, two in each quadrant
FOOD_CELLS: set[tuple[int, int]] = {
    (1, 1), (1, 13),
    (13, 1), (13, 13),
    (1, 7), (13, 7),
    (7, 1), (7, 13),
}

# Water depots — 4 cells, persistent (do not respawn-deplete)
WATER_CELLS: set[tuple[int, int]] = {
    (3, 3), (3, 11),
    (11, 3), (11, 11),
}

# Medicine depots — 2 cells, scarce. Respawn 25 steps after pickup.
MEDICINE_CELLS: set[tuple[int, int]] = {
    (5, 7), (9, 7),
}

# Walls — chokepoints around the safehouse approaches and along the diagonals
WALL_CELLS: set[tuple[int, int]] = {
    (4, 6), (4, 8),     # north of safehouse
    (10, 6), (10, 8),   # south of safehouse
    (6, 4), (8, 4),     # west of safehouse
    (6, 10), (8, 10),   # east of safehouse
    (2, 2), (2, 12),    # NW / NE corner chokes
    (12, 2), (12, 12),  # SW / SE corner chokes
}

# Agent spawns — 5 agents, all start inside the safehouse
AGENT_SPAWNS: list[tuple[int, int]] = [
    (6, 6),  # A0
    (6, 7),  # A1
    (6, 8),  # A2
    (7, 7),  # A3 (centre)
    (8, 7),  # A4
]

# Initial zombie spawns — 3 corners (matches v1 starting count)
ZOMBIE_SPAWNS: list[tuple[int, int]] = [
    (0, 0),
    (0, GRID_COLS - 1),
    (GRID_ROWS - 1, GRID_COLS - 1),
]

# Pool of candidate cells for wave spawns. Filled with non-safehouse,
# non-resource, non-wall cells outside a 4-cell ring around the safehouse.
def _build_wave_pool() -> list[tuple[int, int]]:
    pool: list[tuple[int, int]] = []
    blocked = (
        SAFEHOUSE_CELLS
        | FOOD_CELLS
        | WATER_CELLS
        | MEDICINE_CELLS
        | WALL_CELLS
    )
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r, c) in blocked:
                continue
            # Exclude an outer-edge ring (rows/cols 0 or 14): zombies start
            # there already, plus we want wave zombies to appear *inside* the
            # map so they apply pressure faster.
            if r == 0 or r == GRID_ROWS - 1 or c == 0 or c == GRID_COLS - 1:
                continue
            # Exclude immediate safehouse perimeter (4-cell ring)
            if 5 <= r <= 9 and 5 <= c <= 9:
                continue
            pool.append((r, c))
    return pool


WAVE_SPAWN_POOL: list[tuple[int, int]] = _build_wave_pool()


def build_grid() -> list[list[str]]:
    """Initial base grid with cell-type markers (no agents/zombies)."""
    grid = [["." for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    for r, c in WALL_CELLS:
        grid[r][c] = "#"
    for r, c in FOOD_CELLS:
        grid[r][c] = "F"
    for r, c in WATER_CELLS:
        grid[r][c] = "W"
    for r, c in MEDICINE_CELLS:
        grid[r][c] = "M"
    for r, c in SAFEHOUSE_CELLS:
        grid[r][c] = "S"
    return grid


def render_grid(
    base_grid: list[list[str]],
    agents: list[dict],
    zombies: list[dict],
    food_present: dict[tuple[int, int], bool] | None = None,
    medicine_present: dict[tuple[int, int], bool] | None = None,
) -> list[list[str]]:
    """Render agents/zombies on top of base, also blanking depleted depots.

    `food_present` and `medicine_present` are optional dicts indexed by cell
    coordinate; if a cell maps to False, the depot is currently depleted and
    is rendered as '.' instead of 'F' / 'M'.
    """
    grid = copy.deepcopy(base_grid)

    if food_present is not None:
        for cell, present in food_present.items():
            if not present:
                r, c = cell
                grid[r][c] = "."
    if medicine_present is not None:
        for cell, present in medicine_present.items():
            if not present:
                r, c = cell
                grid[r][c] = "."

    for z in zombies:
        r, c = z["row"], z["col"]
        grid[r][c] = "Z"

    for a in agents:
        if a.get("is_alive", True):
            r, c = a["row"], a["col"]
            grid[r][c] = f"A{a['agent_id']}"

    return grid
