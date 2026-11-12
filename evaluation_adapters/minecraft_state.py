"""The public state contract and objective predicates, shared by trusted runs."""
from functools import lru_cache
import json
import math
from pathlib import Path


RAY_RANGE = 10.0


def item_id(name):
    return str(name).strip().removeprefix("minecraft:").replace(" ", "_")


def objective_satisfied(goal, seen):
    goal = item_id(goal)
    return bool({"log", "log2"} & seen) if goal == "log" else goal in seen


def inventory_slots(observation):
    inventory = observation["inventory"]
    equipment = observation.get("equipment", {}).get("name", ())
    hand = item_id(equipment[0]) if len(equipment) else None
    rows = []
    for slot, (name, quantity) in enumerate(zip(inventory["name"], inventory["quantity"])):
        name = item_id(name)
        if int(quantity) <= 0 or name in {"air", "none", ""}:
            continue
        row = {"slot": slot, "item_id": name, "quantity": int(quantity),
               "equipped": slot == 0 and name == hand}
        for field in ("variant", "cur_durability", "max_durability"):
            row[field] = int(inventory[field][slot]) if field in inventory else None
        rows.append(row)
    return rows


def inventory_items(observation):
    items = {row["item_id"] for row in inventory_slots(observation)}
    changes = observation.get("delta_inv", {})
    for source in ("inc_name_by_craft", "inc_name_by_other"):
        quantities = changes.get(source.replace("_name_", "_quantity_"), ())
        items.update(item_id(name) for name, quantity in zip(changes.get(source, ()), quantities)
                     if int(quantity) > 0 and item_id(name) not in {"air", "none", ""})
    return items


@lru_cache(maxsize=1)
def craft_items():
    directory = Path(__file__).parent
    # Source/image layout and the participant ZIP layout use the same catalogue.
    for path in (directory / "minecraft/action_items.json", directory.parent / "action_items.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))["craft_smelt_items"]
    raise RuntimeError("The fixed Minecraft action catalogue is missing")


def crosshair_hit(observation):
    import numpy as np

    miss = {"block_name": "none_in_range", "block_distance": None,
            "traced_block_x": None, "traced_block_y": None, "traced_block_z": None,
            "harvest_level": None, "is_tool_not_required": None, "range_meters": RAY_RANGE}
    rays = observation.get("rays", {})

    def first(key):
        values = np.asarray(rays.get(key, ())).reshape(-1)
        return values[0].item() if values.size else None

    distance, name = first("block_distance"), first("block_name")
    if (distance is None or not math.isfinite(float(distance)) or not 0 <= float(distance) <= RAY_RANGE
            or name is None or item_id(name) in {"", "air", "none"}):
        return miss
    result = dict(miss, block_name=item_id(name), block_distance=float(distance))
    for field in ("traced_block_x", "traced_block_y", "traced_block_z", "harvest_level"):
        value = first(field)
        if value is not None and math.isfinite(float(value)):
            result[field] = int(value)
    value = first("is_tool_not_required")
    result["is_tool_not_required"] = bool(value) if value is not None else None
    return result


def inventory_change(before, after):
    counts = {}
    for sign, sample in ((-1, before), (1, after)):
        for row in sample["inventory"]:
            name = row["item_id"]
            counts[name] = counts.get(name, 0) + sign * row["quantity"]
    return {key: value for key, value in counts.items() if value}
