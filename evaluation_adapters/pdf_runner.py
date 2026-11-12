"""One fresh world, one Agent process, one independently metered PDF episode."""
import base64
import hashlib
import io
import json
import time
from pathlib import Path

from .environment_events import EnvironmentEvents
from .minecraft_recording import EpisodeRecorder
from .minecraft_state import (craft_items, crosshair_hit, inventory_change,
                              inventory_items, inventory_slots, objective_satisfied)
from .model_gateway import atomic_json
from .pdf_scoring import POINTS, Score
from .torch_placement import VOXEL_SIZE


def validate_scenarios(scenes):
    if not isinstance(scenes, list) or not scenes:
        raise ValueError('场景必须是非空 JSON 对象列表')
    ids = set()
    full_conditions = None
    condition_keys = ('task_id', 'initial_inventory', 'start_time', 'allow_time_passage',
                      'allow_mob_spawn', 'generate_world_type')
    for scene in scenes:
        if not isinstance(scene, dict) or scene.get('mode') not in ('tutorial', 'full'):
            raise ValueError('每个场景须明确 mode: tutorial/full')
        if scene.get('task_id') != 'open-ended' or type(scene.get('world_seed')) not in (int, str):
            raise ValueError('场景须使用 open-ended 和有效种子')
        if not isinstance(scene.get('id'), str) or not scene['id'] or scene['id'] in ids:
            raise ValueError('场景 id 必须唯一且非空')
        ids.add(scene['id'])
        if not isinstance(scene.get('label'), str) or not scene['label']:
            raise ValueError('缺少场景名称')
        if not isinstance(scene.get('goals'), list) or not scene['goals']:
            raise ValueError('缺少场景目标')
        if scene['mode'] == 'tutorial':
            if type(scene.get('max_steps')) is not int or scene['max_steps'] < 1:
                raise ValueError('教学例子须设置练习步数上限')
        else:
            if 'max_steps' in scene or scene.get('initial_inventory', []) != []:
                raise ValueError('完整计分局必须空背包开始；参考步数不能用作终止上限')
            if scene['goals'] != list(POINTS):
                raise ValueError('完整例子须包含 PDF 的全部 11 个里程碑')
            conditions = {k: scene.get(k) for k in condition_keys}
            if full_conditions is not None and conditions != full_conditions:
                raise ValueError('多种子计分局除种子外必须使用相同环境初始条件')
            full_conditions = conditions


def public_observation(observation, noop_action, step=0, step_limit=None,
                       include_image=True, last_action=None):
    import numpy as np
    from PIL import Image
    pixels = np.asarray(observation['rgb'], dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError('MineDojo RGB shape must be (3,H,W)')
    image = io.BytesIO()
    if include_image:
        Image.fromarray(np.moveaxis(pixels, 0, -1)).save(image, format='JPEG', quality=90)
    return {'image_jpeg_base64': base64.b64encode(image.getvalue()).decode('ascii'),
            'inventory': inventory_slots(observation),
            'position': [float(x) for x in observation['location_stats']['pos']],
            'orientation': {k: float(observation['location_stats'].get(k, 0)) for k in ('yaw', 'pitch')},
            'health': float(observation['life_stats']['life']), 'noop_action': list(noop_action),
            'step': step, 'step_limit': step_limit, 'crosshair_hit': crosshair_hit(observation),
            'craftable': [name for name, valid in zip(craft_items(), observation.get('masks', {}).get('craft_smelt', ())) if bool(valid)],
            'nearby_tools': {k: bool(observation.get('nearby_tools', {}).get(k, False)) for k in ('table', 'furnace')},
            'last_action': last_action}


def checked_action(message, env):
    action = message.get('value') if isinstance(message, dict) and message.get('type') == 'action' else None
    limits = [int(v) for v in env.action_space.nvec]
    if (not isinstance(action, list) or len(action) != len(limits) or
            any(type(v) is not int or not 0 <= v < limit for v, limit in zip(action, limits))):
        return [int(v) for v in env.action_space.no_op()], False
    import numpy as np
    if not env.action_space.contains(np.asarray(action, dtype=np.int64)):
        return [int(v) for v in env.action_space.no_op()], False
    return action, True


def create_minedojo_environment(scene):
    import minedojo
    from minedojo.sim import InventoryItem
    options = {k: scene[k] for k in ('generate_world_type', 'start_time', 'allow_mob_spawn',
                                    'allow_time_passage') if k in scene}
    options['initial_inventory'] = [InventoryItem(**v) for v in scene.get('initial_inventory', [])]
    options['seed'] = scene.get('env_seed', int.from_bytes(hashlib.sha256(str(scene['world_seed']).encode()).digest()[:4], 'big'))
    # Native open-ended respawn is retained. Never use FastResetWrapper after death.
    return minedojo.make(task_id='open-ended', image_size=(400, 640), world_seed=scene['world_seed'],
                         fast_reset=False, use_lidar=True, lidar_rays=[(0., 0., 10.)],
                         use_voxel=True, voxel_size=dict(VOXEL_SIZE), **options)


def run_episode(channel, env, scene, budget, directory, *, record=True):
    """The supervisor enforces the deadline even while Agent/env.step is blocked.

    Checkpoints include only fully returned, pre-cutoff env.step calls. An
    interrupted world is settled by the supervisor from its last checkpoint.
    """
    import numpy as np
    directory = Path(directory)
    score = Score()
    seen = set()
    invalid = 0
    full = scene['mode'] == 'full'
    observation = env.reset()  # Exactly once in the entire world session.
    # minedojo.make wraps MineDojoSim in ARNNWrapper; compatible injected
    # environments may expose prev_info directly.
    info = getattr(env, 'prev_info', None)
    if info is None:
        info = env.env.prev_info
    events = EnvironmentEvents(info, observation)
    if full and inventory_items(observation):
        raise RuntimeError('完整计分局的初始背包必须为空')
    if info.get('keepInventory', False):
        raise RuntimeError('keepInventory 必须关闭')
    initial_slots = inventory_slots(observation)
    noop = [int(v) for v in env.action_space.no_op()]
    recorder = EpisodeRecorder(directory, scene.get('max_steps', 36000)) if record else None
    last_action = None
    budget.start()  # Environment initialized; includes Agent import/config/reset and every API/thinking delay.

    def outcome(reason, status='completed'):
        return {'scenario': dict(scene), 'status': status, 'termination': reason,
                'scoring': score.result() if full else None,
                'metrics': {'task_success': bool(score.first['iron_pickaxe'] is not None) if full else
                            all(objective_satisfied(g, seen) for g in scene['goals']),
                            'action_count': score.steps, 'deaths': score.deaths,
                            'invalid_actions': invalid,
                            'execution_seconds': min(3600, time.monotonic() - budget.started)},
                'objectives': [{'id': g, 'completed': objective_satisfied(g, seen)} for g in scene['goals']],
                'api': budget.snapshot(), 'scored': full,
                'note': 'PDF 规则自测' if full else '教学练习，不计入平均分'}

    def checkpoint():
        atomic_json(directory / 'checkpoint.json', outcome('running', 'running'))

    checkpoint()
    try:
        # Lazy channel starts a separate fresh Agent only after this world's clock starts.
        channel.send({'type': 'reset', 'goal': scene['goals']})
        if channel.receive() != {'type': 'ready'}:
            raise ValueError('Agent 未确认 reset')
        if recorder:
            recorder.append(observation['rgb'])
        with (directory / 'steps.jsonl').open('a', encoding='utf-8') as log:
            while True:
                reason = budget.reason()
                if reason:
                    break
                if not full and score.steps >= scene['max_steps']:
                    reason = 'tutorial_step_limit'
                    break
                if not full and all(objective_satisfied(g, seen) for g in scene['goals']):
                    reason = 'tutorial_goals'
                    break
                sample = public_observation(observation, noop, score.steps, scene.get('max_steps'), last_action=last_action)
                channel.send({'type': 'step', 'observation': sample})
                action, valid = checked_action(channel.receive(), env)
                if budget.reason():  # In particular, never execute the action after the cap-hitting LLM call.
                    reason = budget.reason()
                    break
                next_observation, _, done, info = env.step(np.asarray(action, dtype=np.int64))
                if budget.reason():  # An incomplete/late action contributes no events or step.
                    reason = budget.reason()
                    break
                if done:
                    # MineDojo returns the previous observation on bridge failure. Do not award phantom events.
                    raise RuntimeError('MineDojo mission terminated; same-world respawn unavailable or bridge failed')
                facts = events.consume(next_observation, info, action)
                score.complete_step(**facts)
                seen.update(facts['items'])
                invalid += int(not valid)
                after = public_observation(next_observation, noop, score.steps, include_image=False)
                last_action = {'action': action, 'valid': valid, 'inventory_delta': inventory_change(sample, after),
                               'position_delta': [a-b for a, b in zip(after['position'], sample['position'])],
                               'health_delta': after['health']-sample['health']}
                event = {'step': score.steps, 'action': action, 'valid': valid, 'confirmed': facts,
                         'after': after, 'api_tokens': budget.total}
                if events.placement_evidence:
                    event['torch_placement_evidence'] = events.placement_evidence
                log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                checkpoint()
                observation = next_observation
                if recorder:
                    recorder.append(observation['rgb'])
                # No health<=0 break: native respawn continues in exactly the same world.
                if full and score.first['iron_pickaxe'] is not None:
                    reason = 'iron_pickaxe'
                    break
        result = outcome(reason, 'error' if reason == 'accounting_error' else 'completed')
        atomic_json(directory / 'outcome.json', result)
        if recorder:
            recording = recorder.finish()
        else:
            recording = {'capture': 'disabled-by-local-user'}
        atomic_json(directory / 'trajectory.json', {'goals': scene['goals'], 'initial_inventory': initial_slots,
                    'steps_file': 'steps.jsonl', 'recording': recording})
        return result
    finally:
        if recorder:
            recorder.__exit__(RuntimeError, None, None)  # Also cleans up a partially written encoder on error.
