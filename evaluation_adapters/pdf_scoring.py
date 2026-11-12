"""PDF rules, independent of MineDojo, the Agent and model responses.

None in JSON means +infinity (not an achieved milestone). No score rounding.
"""
import math

POINTS = dict(log=2, planks=2, stick=2, crafting_table=4, wooden_pickaxe=5,
              cobblestone=4, stone_pickaxe=6, furnace=5, iron_ore=6,
              iron_ingot=7, iron_pickaxe=12)
SIDE = dict(coal=2, torch=1, placed_torch=2)
SECONDS = 3600
TOKENS = 5_000_000


class Score:
    def __init__(self):
        self.steps = 0
        self.first = {name: None for name in (*POINTS, *SIDE)}
        self.deaths = 0

    def complete_step(self, items=(), *, deaths=0, placed_torch=False):
        """Call exactly once per completed bottom-level env.step, even for noops.

        Items/events must come from the trusted environment, never the Agent.
        Process the whole terminal step before checking iron_pickaxe.
        """
        if self.first['iron_pickaxe'] is not None:
            raise ValueError('episode already finished')
        if type(deaths) is not int or deaths < 0:
            raise ValueError('invalid environment death count')
        self.steps += 1
        self.deaths += deaths
        for name in items:
            name = str(name).removeprefix('minecraft:').replace(' ', '_')
            if name == 'log2':
                name = 'log'
            if name in self.first and name != 'placed_torch' and self.first[name] is None:
                self.first[name] = self.steps
        if placed_torch and self.first['placed_torch'] is None:
            self.first['placed_torch'] = self.steps

    def result(self, *, cheated=False):
        t = {key: math.inf if value is None else value for key, value in self.first.items()}
        iron = t['iron_pickaxe']
        stone = min(max(t['stone_pickaxe'], t['furnace']), iron)
        wood = min(t['wooden_pickaxe'], stone)
        tech = 55 if math.isfinite(iron) else sum(p for k, p in POINTS.items() if math.isfinite(t[k]))
        side = sum(p for k, p in SIDE.items() if math.isfinite(t[k]))
        efficiency = sum(w * b / (b + c) for w, b, c in
                         ((8, 1500, wood), (12, 6000, stone), (20, 36000, iron)))
        penalty = min(2 * self.deaths, 10)
        raw = min(max(tech + side + efficiency - penalty, 0), 100)
        return {'steps': self.steps, 'first_acquired': dict(self.first),
                'checkpoints': {k: None if math.isinf(v) else v for k, v in
                                (('wood', wood), ('stone', stone), ('iron', iron))},
                'supplemented': [k for k in POINTS if self.first[k] is None] if tech == 55 else [],
                'tech': tech, 'side': side, 'efficiency': efficiency,
                'deaths': self.deaths, 'death_penalty': penalty,
                'cheated': bool(cheated), 'score_before_adjudication': raw,
                'score': 0 if cheated else raw}


def aggregate(episodes):
    """Do not silently omit failed/missing full worlds or include tutorials."""
    full = [e for e in episodes if e['scenario']['mode'] == 'full']
    valid = bool(full) and all(e.get('status') == 'completed' and e.get('scoring') for e in full)
    return {'full_worlds': len(full), 'eligible': valid,
            'mean_score': sum(e['scoring']['score'] for e in full) / len(full) if valid else None,
            'notice': '本地规则自测分；正式成绩以组织方隐藏种子及平台核验为准。'}
