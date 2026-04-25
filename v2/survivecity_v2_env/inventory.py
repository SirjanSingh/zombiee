"""Per-agent inventory bookkeeping.

Pure functions over a list-of-strings inventory. Items are 'food' | 'water'
| 'medicine'. Cap of 3 slots per agent. No stacking.

The actual mutation of agent state is done by game.py — this module provides
the bookkeeping primitives so game.py stays terse.
"""

from __future__ import annotations

from typing import Literal, Optional

INVENTORY_CAP = 3
ItemType = Literal["food", "water", "medicine"]


def has_free_slot(inventory: list[str]) -> bool:
    return len(inventory) < INVENTORY_CAP


def add_item(inventory: list[str], item: str) -> bool:
    """Append an item if there's room. Returns True on success, False if full."""
    if not has_free_slot(inventory):
        return False
    inventory.append(item)
    return True


def remove_at(inventory: list[str], slot: int) -> Optional[str]:
    """Remove and return the item at the given slot index, or None if invalid."""
    if slot is None:
        return None
    if not (0 <= slot < len(inventory)):
        return None
    return inventory.pop(slot)


def remove_first(inventory: list[str], item: str) -> bool:
    """Remove the first occurrence of `item` from the inventory.

    Returns True if removed, False if no such item.
    """
    for i, it in enumerate(inventory):
        if it == item:
            inventory.pop(i)
            return True
    return False


def find_first_slot(inventory: list[str], item: str) -> Optional[int]:
    for i, it in enumerate(inventory):
        if it == item:
            return i
    return None


def occupancy(inventory: list[str]) -> int:
    return len(inventory)


def empty_slots(inventory: list[str]) -> int:
    return INVENTORY_CAP - len(inventory)
