"""Translate environment observations/raw info, never Agent claims, into facts."""
from .minecraft_state import inventory_items, item_id
from .torch_placement import TorchPlacement


def count(info, *keys):
    value = info
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or int(value) != value:
        return None
    return int(value)


def native_counters(info):
    # JSONWorldDataHelper: stat.useItem.minecraft.torch -> stat/use_item/minecraft/torch.
    return count(info, 'stat', 'deaths'), count(info, 'stat', 'use_item', 'minecraft', 'torch')


class EnvironmentEvents:
    def __init__(self, initial_info, initial_observation=None):
        self.previous_deaths, self.previous_torches = native_counters(initial_info)
        self.event_deaths = 0
        self.initial_deaths = self.previous_deaths
        self.accounted_deaths = 0
        self.placements = TorchPlacement(initial_observation) if initial_observation is not None else None
        self.placement_evidence = None

    def consume(self, observation, info, action=None):
        death_total, torch_total = native_counters(info)
        if 'living_death_event_fired' not in info and death_total is None:
            raise RuntimeError('环境未提供死亡事件或累计死亡计数，不能可靠计分')
        if torch_total is None and self.placements is None:
            raise RuntimeError('环境未提供普通火把成功使用统计，不能可靠确认放置')
        # Native event is an edge, reset by the observation producer after each sample.
        # The cumulative stat can arrive later; max avoids counting it twice.
        self.event_deaths += int(bool(info.get('living_death_event_fired', False)))
        stat_deaths = 0
        if death_total is not None:
            if self.initial_deaths is None:
                raise RuntimeError('缺失初始死亡计数，不能建立环境统计基线')
            if death_total < self.initial_deaths:
                raise RuntimeError('死亡统计回退；禁止世界重置或回滚')
            stat_deaths = death_total - self.initial_deaths
        confirmed = max(self.event_deaths, stat_deaths)
        deaths = confirmed - self.accounted_deaths
        self.accounted_deaths = confirmed
        native_placed = False
        self.placement_evidence = None
        if torch_total is not None:
            if self.previous_torches is None:
                if self.placements is None:
                    raise RuntimeError('缺失或回退的火把放置统计基线')
            elif torch_total < self.previous_torches:
                raise RuntimeError('缺失或回退的火把放置统计基线')
            else:
                native_placed = torch_total > self.previous_torches
                if native_placed:
                    self.placement_evidence = {'source': 'native_use_stat',
                                               'before': self.previous_torches, 'after': torch_total}
        world_evidence = self.placements.consume(observation, action) if self.placements else None
        if world_evidence:
            self.placement_evidence = world_evidence
        placed = native_placed or world_evidence is not None
        self.previous_torches = torch_total
        items = inventory_items(observation)
        equipment = observation.get('equipment', {})
        for name, quantity in zip(equipment.get('name', ()), equipment.get('quantity', ())):
            if quantity > 0 and item_id(name) not in {'air', 'none', ''}:
                items.add(item_id(name))
        return {'items': sorted(items), 'deaths': deaths, 'placed_torch': placed}
