"""Confirm native place commands from trusted inventory and world transitions.

MineDojo 0.1's PlaceCommandsImplementation sets a block and consumes its item
without incrementing stat.useItem. A quantity decrease alone is insufficient:
require the executed torch-place action and a newly created world torch too.
The grid is controller-only; it is never added to the Agent observation.
"""
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np

from .minecraft_state import inventory_slots


# Cover native block reach, the player's eye height, and the adjacent block.
VOXEL_SIZE = dict(xmin=-6, xmax=6, ymin=-6, ymax=8, zmin=-6, zmax=6)
BlockPosition = tuple[int, int, int]


@dataclass(frozen=True)
class WorldSnapshot:
    origin: BlockPosition
    shape: tuple[int, int, int]
    torches: frozenset[BlockPosition]
    slots: dict[int, str]
    quantity: int

    def contains(self, position: BlockPosition) -> bool:
        return all(start <= value < start + size
                   for value, start, size in zip(position, self.origin, self.shape))


def snapshot(observation: dict) -> WorldSnapshot:
    """Map relative grid cells to absolute block positions, including negatives."""
    try:
        blocks = np.asarray(observation['voxels']['block_name'])
        position = np.asarray(observation['location_stats']['pos'], dtype=float)
        expected = tuple(VOXEL_SIZE[f'{axis}max'] - VOXEL_SIZE[f'{axis}min'] + 1
                         for axis in 'xyz')
        if blocks.shape != expected or position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('invalid world grid or position')
        # Java BlockPos(double, double, double) floors each coordinate.
        origin = tuple(math.floor(value) + VOXEL_SIZE[f'{axis}min']
                       for axis, value in zip('xyz', position))
        torches = frozenset(tuple(start + int(offset) for start, offset in zip(origin, cell))
                            for cell in np.argwhere((blocks == 'torch') | (blocks == 'minecraft:torch')))
        rows = inventory_slots(observation)
        slots = {row['slot']: row['item_id'] for row in rows}
        quantity = sum(row['quantity'] for row in rows if row['item_id'] == 'torch')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError('缺少有效的火把放置核验网格或背包；请使用完整新版开发包') from exc
    return WorldSnapshot(origin, blocks.shape, torches, slots, quantity)


class TorchPlacement:
    def __init__(self, observation: dict):
        self.previous = snapshot(observation)

    def consume(self, observation: dict, action: Optional[Sequence[int]]) -> Optional[dict]:
        before, after = self.previous, snapshot(observation)
        self.previous = after
        if action is None or len(action) != 8 or action[5] != 6:
            return None
        selected = before.slots.get(int(action[7]))
        if selected != 'torch' or after.quantity >= before.quantity:
            return None
        # An existing torch entering the moving grid is not a new placement.
        created = sorted(position for position in after.torches - before.torches
                         if before.contains(position))
        if not created:
            return None
        return {'source': 'world_transition', 'position': list(created[0]),
                'inventory_before': before.quantity, 'inventory_after': after.quantity}
