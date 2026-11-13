"""Feedback-driven MineDojo policy using only stdlib and bundled helpers."""
from collections import Counter, deque
import json
import math
import time

from minecraft_actions import craft_action, slot_action
from model_api import ModelClient, ModelResponseError


LIMITS = (3, 3, 4, 25, 25, 8, 244, 36)
RECIPES = {
    "planks": (4, {"log": 1}, None),
    "stick": (4, {"planks": 2}, None),
    "crafting_table": (1, {"planks": 4}, None),
    "wooden_pickaxe": (1, {"planks": 3, "stick": 2}, "table"),
    "stone_pickaxe": (1, {"cobblestone": 3, "stick": 2}, "table"),
    "furnace": (1, {"cobblestone": 8}, "table"),
    "iron_ingot": (1, {"iron_ore": 1}, "furnace"),
    "iron_pickaxe": (1, {"iron_ingot": 3, "stick": 2}, "table"),
}
FACILITIES = {"table": "crafting_table", "furnace": "furnace"}
RESOURCE_BLOCKS = {"log": {"log", "log2", "wood"},
                   "cobblestone": {"stone", "cobblestone"},
                   "coal": {"coal_ore"}, "iron_ore": {"iron_ore"}}
DEFAULTS = {
    "max_output_tokens": 1024, "model_timeout": 45,
    "model_min_interval": 12, "max_action_steps": 16,
    "attack_timeout": 180, "stuck_steps": 12,
    "operation_wait_steps": 2, "smelt_wait_steps": 24,
    "operation_retry_limit": 3, "model_error_limit": 3,
    "model_backoff_steps": 24, "model_backoff_seconds": 3,
    "episode_seconds": 3600, "model_token_budget": 4500000,
    "interaction_distance": 3.0, "memory_size": 48,
    "low_health": 6, "log_decisions": True,
    "progress_interval": 100, "search_timeout": 120,
    "collect_steps": 64, "budget_reserve_seconds": 5,
    "token_input_reserve": 16000, "mining_cooldown": 30,
    "target_lifetime": 160, "route_steps": 48,
    "navigation_timeout": 24, "collect_arrival_steps": 8,
    "facility_return_steps": 100, "recovery_steps": 6,
    "visual_move_steps": 8, "model_control_steps": 16,
    "navigation_review_steps": 16, "recovery_failure_limit": 2,
    "level_pitch": 45, "max_move_pitch": 55,
}


def item_name(value):
    name = str(value).strip().removeprefix("minecraft:").replace(" ", "_")
    if name == "log2" or name.endswith("_log"):
        return "log"
    if name.endswith("_planks"):
        return "planks"
    return name


def resource_matches(resource, block):
    name = str(block).strip().removeprefix("minecraft:").replace(" ", "_")
    return name in RESOURCE_BLOCKS.get(resource, set()) or (
        resource == "log" and name.endswith("_log"))


def valid_action(action):
    return (isinstance(action, list) and len(action) == 8
            and all(type(v) is int and 0 <= v < limit
                    for v, limit in zip(action, LIMITS)))


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class Agent:
    def configure(self, config):
        self.settings = dict(DEFAULTS)
        self.settings.update(config["agent"])
        for key, default in DEFAULTS.items():
            value = self.settings[key]
            if isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(key + " must be boolean")
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or value <= 0):
                raise ValueError(key + " must be positive and finite")
            elif isinstance(default, int) and type(value) is not int:
                raise ValueError(key + " must be an integer")
        if self.settings["level_pitch"] >= self.settings["max_move_pitch"]:
            raise ValueError("level_pitch must be below max_move_pitch")
        self.client = ModelClient(config=config,
                                  max_output_tokens=self.settings["max_output_tokens"],
                                  timeout=self.settings["model_timeout"])
        self.reset([])

    def reset(self, goal):
        self.goals = [item_name(g) for g in goal]
        self.seen = set()
        self.node = "observe"
        self.task = None
        self.previous = None
        self.last_action = None
        self.pending = None
        self.segment = None
        self.failures = Counter()
        self.blocked_until = {}
        self.memory = deque(maxlen=self.settings["memory_size"])
        self.recent = deque(maxlen=self.settings["stuck_steps"])
        self.facilities = {}
        self.calls = self.tokens = self.model_errors = 0
        self.next_model_step = 0
        self.next_model_time = 0
        self.started = time.monotonic()
        self.attack_key = None
        self.attack_steps = 0
        self.mining_cooldown = {}
        self.terminal_model_error = False
        self.recovery = 0
        self.recovery_remaining = 0
        self.recovery_requested = False
        self.feedback = "new episode"
        self.phase = "search"
        self.collect = None
        self.metrics = Counter()
        self.last_resource_step = 0
        self.last_navigation_step = 0
        self.navigation_anchor = None
        self.task_started_step = 0
        self.phase_started_step = 0
        self.last_recovery_step = 0
        self.target = None
        self.route = None
        self.recovery_origin = None
        self.recovery_completed = False
        self.recovery_measure_after = 0
        self.recovery_failures = 0
        self.failed_trees = {}
        self.visual_move = None
        self.model_control_until = 0
        self.handoff = None
        self.alignment_pending = False
        self.search_replan = False
        self.decision_errors = 0
        self.recovery_hold_until = 0
        self.last_navigation_review = 0
        self.search_origin = None
        self.search_started = 0
        self.recovery_level = 0
        self.wood_requirements = {}
        self.last_summary = -self.settings["progress_interval"]
        self.last_budget_notice = None

    def _log(self, event, **fields):
        if self.settings["log_decisions"]:
            print(json.dumps({"agent": event, **fields}, ensure_ascii=False), flush=True)

    def _observe(self, obs):
        self.node = "observe"
        rows = [dict(row, name=item_name(row["item_id"]))
                for row in obs.get("inventory", [])]
        counts = Counter()
        for row in rows:
            counts[row["name"]] += row["quantity"]
        self.seen.update(name for name, count in counts.items() if count > 0)
        hit = obs.get("crosshair_hit") or {}
        state = {"rows": rows, "counts": counts,
                 "pos": tuple(obs.get("position", (0, 0, 0))),
                 "health": obs.get("health", 20), "step": obs.get("step", 0),
                 "hit": hit, "block": item_name(hit.get("block_name", "none")),
                 "near": obs.get("nearby_tools") or {},
                 "orientation": obs.get("orientation") or {}}
        self.recent.append((state["pos"], dict(counts)))
        for facility in FACILITIES:
            if state["near"].get(facility):
                known = self.facilities.setdefault(facility, {
                    "position": state["pos"], "source": "verified_approach"})
                known["step"] = state["step"]
                if state["block"] == FACILITIES[facility]:
                    block_pos = [hit.get("traced_block_" + a) for a in "xyz"]
                    if all(v is not None for v in block_pos):
                        known["block_position"] = block_pos
        if state["block"] not in {"air", "none", "none_in_range"}:
            location = [hit.get("traced_block_" + axis) for axis in "xyz"]
            record = {"block": state["block"], "position": location}
            if all(v is not None for v in location) and record not in self.memory:
                self.memory.append(record)
        return state

    def _slot(self, state, name):
        return next((row for row in state["rows"] if row["name"] == name), None)

    def _verify(self, state, obs):
        self.node = "verify"
        previous = self.previous
        if self.recovery_completed and state["step"] >= self.recovery_measure_after:
            displacement = distance(self.recovery_origin, state["pos"])
            self._log("recovery_finished", step=state["step"],
                      displacement=round(displacement, 2), level=self.recovery_level)
            if displacement >= 0.75:
                self.recovery_level = 0
                self.recovery_failures = 0
            else:
                self.recovery_failures += 1
                if self.recovery_failures >= self.settings["recovery_failure_limit"]:
                    self.route = None
                    self._handoff(state, "repeated recovery produced negligible displacement; choose a new route")
            self.recovery_completed = False
        if previous:
            moved = distance(state["pos"], previous["pos"])
            changed = state["counts"] != previous["counts"]
            if any(quantity > previous["counts"][name]
                   for name, quantity in state["counts"].items()):
                self.last_resource_step = state["step"]
                self.recovery_level = 0
            if state["counts"]["log"] > previous["counts"]["log"]:
                self.target = self.route = None
                self.metrics["log_received"] += state["counts"]["log"] - previous["counts"]["log"]
                self._log("resource_received", step=state["step"], resource="log",
                          quantity=state["counts"]["log"])
            if self.navigation_anchor is None:
                self.navigation_anchor = state["pos"]
            if distance(state["pos"], self.navigation_anchor) >= 0.75:
                self.last_navigation_step = state["step"]
                self.navigation_anchor = state["pos"]
                self.metrics["navigation_progress"] += 1
            if changed:
                self.segment = None
                self.attack_steps = 0
                self.attack_key = None
            if self.collect and state["counts"][self.collect["item"]] > self.collect["before"]:
                self._finish_collect("resource_received", state)
            if (not changed and self.last_action and self.last_action[5] == 3
                    and self.task and self.task["kind"] == "gather" and self.attack_key
                    and resource_matches(self.task["item"], previous["block"])):
                old_pos = self.attack_key[1:]
                new_pos = tuple(state["hit"].get("traced_block_" + a) for a in "xyz")
                if (not resource_matches(self.task["item"], state["block"])
                        or (all(v is not None for v in old_pos + new_pos) and old_pos != new_pos)):
                    self.collect = {"position": old_pos, "start": state["step"],
                                    "item": self.task["item"],
                                    "before": state["counts"][self.task["item"]],
                                    "arrived": None}
                    self._phase("collect", state)
                    self._log("collect_started", step=state["step"], target=old_pos,
                              resource=self.task["item"], evidence="crosshair_changed")
                    self.target = None
                    self.attack_key = None
                    self.segment = None
                    self.feedback = "target changed after mining; check nearby drops"
            if self._move_hazard(state):
                self._invalidate_path(state, "hazard_feedback")
            if state["health"] < previous["health"]:
                self.segment = None
                self.feedback = "health decreased; avoid hazards"
            lost = sum(state["counts"].values()) < sum(previous["counts"].values())
            if moved > 12 or (lost and previous["health"] <= 0 < state["health"]):
                self.pending = self.segment = None
                self.collect = None
                self.target = self.route = None
                self._invalidate_path(state, "state_discontinuity")
                self.facilities.clear()
                self.attack_key = None
                self.recent.clear()
                self.feedback = "possible respawn or displacement; recheck resources"
                self._log("state_discontinuity", step=state["step"])
        last = obs.get("last_action") or {}
        if last.get("valid") is False:
            self._invalidate_path(state, "action_rejected")
            self.segment = None
            self.feedback = "previous action rejected"
        if not self.pending:
            return
        pending = self.pending
        kind, name = pending["kind"], pending["item"]
        if kind == "craft":
            success = state["counts"][name] > pending["before"]
        elif kind == "equip":
            success = any(row["name"] == name and row.get("equipped")
                          for row in state["rows"])
        else:
            success = bool(state["near"].get(name))
        if success:
            self.failures.pop(pending["key"], None)
            self.blocked_until.pop(pending["key"], None)
            self.feedback = kind + " confirmed: " + name
            self._log("confirmed", operation=kind, item=name, step=state["step"])
            self.pending = None
        elif state["step"] >= pending["deadline"]:
            key = pending["key"]
            self.failures[key] += 1
            delay = min(240, 12 * self.failures[key])
            self.blocked_until[key] = state["step"] + delay
            if self.failures[key] >= self.settings["operation_retry_limit"]:
                self.recovery_remaining = 0
                self.feedback = "repeated operation failure; reposition before retry"
                self.recovery_requested = True
            self.feedback = kind + " unconfirmed: " + name + "; check prerequisites and positioning"
            self.pending = self.segment = None

    def _need(self, item, amount, state, trail=()):
        counts = state["counts"]
        if counts[item] >= amount:
            return None
        if item in trail:
            return {"kind": "search", "item": item, "amount": amount}
        trail = (*trail, item)
        if item not in RECIPES:
            required = "stone_pickaxe" if item == "iron_ore" else "wooden_pickaxe"
            if item in {"cobblestone", "coal", "iron_ore"}:
                usable = ("iron_pickaxe", "stone_pickaxe") if item == "iron_ore" else (
                    "iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
                if not any(counts[tool] for tool in usable):
                    return self._need(required, 1, state, trail)
            return {"kind": "gather", "item": item, "amount": amount}
        output, ingredients, facility = RECIPES[item]
        if facility and not state["near"].get(facility):
            facility_item = FACILITIES[facility]
            if counts[facility_item]:
                return {"kind": "place", "item": facility}
            if facility in self.facilities:
                known = self.facilities[facility]
                if state["step"] - known["step"] < 400:
                    return {"kind": "return", "item": facility,
                            "position": known["position"]}
            needed = self._need(facility_item, 1, state, trail)
            if needed:
                return needed
        batches = math.ceil((amount - counts[item]) / output)
        for ingredient, quantity in ingredients.items():
            needed = self._need(ingredient, quantity * batches, state, trail)
            if needed:
                return needed
        if item == "iron_ingot":
            # The bridge owns smelting; try its advertised recipe before demanding fuel.
            hinted = "iron_ingot" in self.observation.get("craftable", [])
            failed = self.failures["craft:iron_ingot"] > 0
            if (failed or not hinted) and not (counts["coal"] or counts["charcoal"]):
                return self._need("coal", 1, state, trail)
        return {"kind": "craft", "item": item, "amount": amount}

    def _invalidate_path(self, state, reason):
        if self.visual_move:
            self._log("path_expired", step=state["step"], reason=reason)
        self.visual_move = None

    def _handoff(self, state, reason):
        if self.terminal_model_error or self.last_budget_notice in {"time_reserve", "token_reserve"}:
            self.feedback = "planning unavailable: " + reason
            self.recovery_requested = False
            self.recovery_remaining = 0
            return
        if self.handoff:
            self.recovery_requested = False
            self.recovery_remaining = 0
            return
        self.segment = None
        self._invalidate_path(state, "model_handoff")
        self.recovery_requested = False
        self.recovery_remaining = 0
        self.handoff = {"start": state["step"], "reason": reason,
                        "status": "waiting_request"}
        self.feedback = reason
        self.metrics["model_handoffs"] += 1
        self._log("model_handoff", step=state["step"], reason=reason,
                  next_request_step=self.next_model_step)

    def _model_controls(self, state):
        return self.handoff is not None or state["step"] < self.model_control_until

    def _phase(self, name, state):
        if name != self.phase:
            self.phase = name
            self.phase_started_step = state["step"]

    def _wood_plan(self, state):
        counts = state["counts"]
        if counts["wooden_pickaxe"]:
            return None
        table_known = (state["near"].get("table") or counts["crafting_table"]
                       or "table" in self.facilities)
        stick_batches = math.ceil(max(0, 2 - counts["stick"]) / 4)
        needed_planks = 3 + 2 * stick_batches + (0 if table_known else 4)
        needed_logs = math.ceil(max(0, needed_planks - counts["planks"]) / 4)
        self.wood_requirements = {"planks_needed": needed_planks,
                                  "logs_needed": needed_logs,
                                  "logs_missing": max(0, needed_logs - counts["log"])}
        if counts["log"] < needed_logs:
            return {"kind": "gather", "item": "log", "amount": needed_logs}
        if counts["planks"] < needed_planks:
            return {"kind": "craft", "item": "planks"}
        if counts["stick"] < 2:
            return {"kind": "craft", "item": "stick"}
        if not table_known:
            return {"kind": "craft", "item": "crafting_table"}
        if not state["near"].get("table"):
            if counts["crafting_table"]:
                return {"kind": "place", "item": "table"}
            return {"kind": "return", "item": "table",
                    "position": self.facilities["table"]["position"]}
        return {"kind": "craft", "item": "wooden_pickaxe"}

    def _plan(self, state):
        self.node = "plan"
        if "wooden_pickaxe" in self.goals and "wooden_pickaxe" not in self.seen:
            task = self._wood_plan(state)
            if task:
                return task
        for goal in self.goals:
            if goal not in self.seen:
                task = self._need(goal, 1, state)
                if task:
                    return task
        return {"kind": "done", "item": "all_goals"}

    def _issue(self, action, kind, item, state):
        wait = (self.settings["smelt_wait_steps"] if item == "iron_ingot"
                else self.settings["operation_wait_steps"])
        self.pending = {"kind": kind, "item": item, "key": kind + ":" + item,
                        "before": state["counts"][item],
                        "deadline": state["step"] + wait}
        self.segment = None
        return action

    def _blocked(self, key, state):
        return state["step"] < self.blocked_until.get(key, 0)

    def _tool(self, state):
        if self.task["kind"] != "gather" or self.task["item"] not in {
                "cobblestone", "coal", "iron_ore"}:
            return None
        names = ("iron_pickaxe", "stone_pickaxe", "wooden_pickaxe")
        if self.task["item"] == "iron_ore":
            names = names[:2]
        for name in names:
            row = self._slot(state, name)
            if row and row.get("cur_durability") != 0:
                return row
        return None

    def _execute(self, state, obs):
        self.node = "execute"
        task = self.task
        kind, name = task["kind"], task["item"]
        if kind != "gather":
            self._phase({"craft": "craft", "place": "place_table", "return": "return_table",
                         "done": "done"}.get(kind, kind), state)
        if kind == "return":
            if state["step"] - self.task_started_step >= self.settings["facility_return_steps"]:
                self.facilities.pop(name, None)
                self.segment = None
                self.feedback = "facility return expired; rebuild or locate again"
                self._log("facility_expired", step=state["step"], facility=name)
                return list(obs["noop_action"])
            return self._navigate(task["position"], state, obs, block_center=False)
        if kind == "done":
            return list(obs["noop_action"])
        if kind == "craft" and not self._blocked("craft:" + name, state):
            if self.failures["craft:" + name] >= self.settings["operation_retry_limit"]:
                self.failures["craft:" + name] = 0
                self.feedback = "reconsidering failed crafting prerequisites"
            return self._issue(craft_action(obs, name), "craft", name, state)
        tool = self._tool(state)
        if tool and not tool.get("equipped") and not self._blocked("equip:" + tool["name"], state):
            if self.failures["equip:" + tool["name"]] >= self.settings["operation_retry_limit"]:
                self.failures["equip:" + tool["name"]] = 0
                self.feedback = "reconsidering failed equipment"
            return self._issue(slot_action(obs, "equip", tool["slot"]),
                               "equip", tool["name"], state)
        hit_distance = state["hit"].get("block_distance")
        in_reach = (isinstance(hit_distance, (int, float))
                    and hit_distance <= self.settings["interaction_distance"])
        if kind == "gather" and resource_matches(name, state["block"]) and in_reach:
            if tool and not tool.get("equipped"):
                return None
            key = tuple(state["hit"].get("traced_block_" + a) for a in "xyz")
            if name == "log" and self.failed_trees.get(key, 0) > state["step"]:
                return None
            key = (state["block"], *key)
            if state["step"] < self.mining_cooldown.get(key, 0):
                self.feedback = "mining target cooling down; reposition"
                return None
            if key != self.attack_key:
                self.attack_key, self.attack_steps = key, 0
            self.attack_steps += 1
            if self.attack_steps <= self.settings["attack_timeout"]:
                action = list(obs["noop_action"])
                action[5] = 3
                self._phase("mine", state)
                self.metrics["attack_rule"] += 1
                return action
            self.feedback = "mining timeout; check tool, reach and target"
            self.mining_cooldown[key] = state["step"] + self.settings["mining_cooldown"]
            self.attack_key = None
            self.attack_steps = 0
            self.segment = None
            self.recovery_requested = True
            self._log("mining_timeout", step=state["step"], block=state["block"], distance=hit_distance)
        elif kind == "gather" and not self.target:
            self._phase("approach" if resource_matches(name, state["block"])
                        else "search", state)
        return None

    def _stuck(self, state):
        if len(self.recent) < self.recent.maxlen or not self.last_action:
            return False
        moving = self.last_action[0] != 0 or self.last_action[1] != 0
        if not moving:
            return False
        position, counts = self.recent[0]
        return distance(position, state["pos"]) < 0.2 and counts == dict(state["counts"])

    def _recover(self, state, obs):
        self.node = "recover"
        self._phase("recover", state)
        leveled = self._level_view(state, obs)
        if leveled is not None:
            self.last_recovery_step = state["step"]
            return leveled
        self.recovery_requested = False
        action = list(obs["noop_action"])
        if not self.recovery_remaining:
            self.recovery += 1
            self.recovery_level = min(3, self.recovery_level + 1)
            self.recovery_remaining = self.settings["recovery_steps"]
            self.recovery_origin = state["pos"]
            self.last_recovery_step = state["step"]
            self.memory.append({"failed_position": state["pos"], "step": state["step"]})
            self.segment = self.route = None
            self.recent.clear()
            if self.recovery_level >= 2:
                self.target = None
            self._log("recovery_started", step=state["step"], level=self.recovery_level)
        if not self._safe_move(state):
            self._handoff(state, "recovery path unconfirmed; inspect image and select a safe short route")
            return action
        if self.visual_move:
            action[0], action[1] = self.visual_move["motion"]
            action[2] = 2
        else:
            self.recovery_remaining = 0
            return self._fallback(state, obs)
        if self.recovery_remaining == self.settings["recovery_steps"]:
            action = list(obs["noop_action"])
            action[4] = (12 + self.recovery_level) if self.recovery % 2 else (12 - self.recovery_level)
            self._handoff(state, "recovery changed view; inspect the new direction before moving")
            self.recovery_completed = True
            self.recovery_measure_after = self.model_control_until
            self.metrics["recovery_steps"] += 1
            return action
        self.recovery_remaining -= 1
        self.metrics["recovery_steps"] += 1
        if not self.recovery_remaining:
            self.recovery_completed = True
            self.feedback = "route changed after recovery; choose a visible tree or new search direction"
        return action

    def _context(self, state, obs):
        return {"goals": self.goals, "observed_milestones": sorted(self.seen),
                "task": self.task, "phase": self.phase, "collection": self.collect, "feedback": self.feedback,
                "inventory": obs.get("inventory", []),
                "position": state["pos"], "orientation": obs.get("orientation"),
                "health": state["health"], "step": state["step"],
                "step_limit": obs.get("step_limit"), "crosshair_hit": state["hit"],
                "nearby_tools": state["near"], "craftable": obs.get("craftable", []),
                "last_action": obs.get("last_action"),
                "material_requirements": self.wood_requirements,
                "tree_target": self.target, "search_route": self.route,
                "model_control_until": self.model_control_until,
                "recovery_failures": self.recovery_failures,
                "search_displacement": (distance(self.search_origin, state["pos"])
                                        if self.search_origin else 0),
                "failed_trees": [{"position": pos, "until": until}
                                 for pos, until in self.failed_trees.items()
                                 if until > state["step"]],
                "task_age": state["step"] - self.task_started_step,
                "phase_age": state["step"] - self.phase_started_step,
                "resource_memory": list(self.memory)[-12:], "facilities": self.facilities,
                "failures": dict(self.failures.most_common(8)),
                "calls": self.calls, "reported_tokens": self.tokens}

    def _ask_model(self, state, obs):
        self.node = "model"
        now = time.monotonic()
        remaining = self.settings["episode_seconds"] - (now - self.started)
        reason = ("time_reserve" if remaining <= self.settings["budget_reserve_seconds"] else
                  "token_reserve" if self.tokens + self.settings["token_input_reserve"] + self.settings["max_output_tokens"] >= self.settings["model_token_budget"] else
                  "terminal_error" if self.terminal_model_error else None)
        if reason:
            if self.last_budget_notice != reason:
                self._log("planning_limited", step=state["step"], reason=reason)
                self.last_budget_notice = reason
            if self.handoff:
                self._log("model_handoff_closed", step=state["step"], reason=reason)
                self.handoff = None
                self.model_control_until = state["step"]
                self.recovery_hold_until = state["step"]
            return False
        if state["step"] < self.next_model_step or now < self.next_model_time:
            if self.handoff:
                self.handoff["status"] = "waiting_request"
            return False
        if self.handoff:
            self.handoff["status"] = "waiting_decision"
        self.client.timeout = min(self.settings["model_timeout"], remaining - self.settings["budget_reserve_seconds"])
        self.client.max_output_tokens = min(self.settings["max_output_tokens"], self.settings["model_token_budget"] - self.tokens)
        prompt = (
            "Control MineDojo. Complete the task using the image and state; return only compact JSON, no explanation. "
            "Schema: {\"action\":[8 integers],\"steps\":integer,\"safe_path\":boolean}. "
            "Action: forward 0/1/2=still/forward/back; strafe 0/1/2=still/left/right; "
            "stance 0/1/2/3=normal/jump/sneak/sprint; pitch,yaw 0..24 with12 neutral; "
            "operation 0/1/3=noop/use/attack,6=place only for place task; craft index0..243; slot0..35. "
            "Use supplied noop. Rules craft/equip. Do not drop/destroy. "
            "Rotate separately, then move with neutral camera. safe_path=true only if image shows support along that short path. "
            "Avoid liquids, cliffs, digging down. Search visible trunks, aim then approach; wood means log. "
            "Maintain search direction, change region after failure, avoid failed_trees. "
            "Attack only the requested reachable resource. During collection approach saved position, do not attack. "
            "Gather material deficit at one tree before crafting. Return to verified facility approach point when requested. "
            "Place uses the actual facility slot, reachable solid support and free space. "
            "Rotation runs once; movement repeats up to "
            + str(self.settings["max_action_steps"]) + " steps. State:\n"
            + json.dumps({"noop": obs["noop_action"], **self._context(state, obs)}, ensure_ascii=False)
        )
        self.calls += 1
        self.next_model_step = state["step"] + self.settings["model_min_interval"]
        stage = "request"
        request_started = time.monotonic()
        try:
            image = obs.get("image_jpeg_base64")
            reply = self.client.chat(prompt, images=[image] if image else [])
            stage = "parse"
            text = reply.strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            decision = json.loads(text)
            if not isinstance(decision, dict):
                raise ValueError("decision must be an object")
            stage = "validation"
            action, steps = decision.get("action"), decision.get("steps")
            if not valid_action(action) or type(steps) is not int or steps < 1:
                raise ValueError("invalid action or duration")
            proposed_attack = action[5] == 3
            if action[5] not in (0, 1, 3, 6):
                raise ValueError("operation reserved for deterministic executor")
            if action[5] == 6:
                facility = self.task["item"]
                row = self._slot(state, FACILITIES.get(facility, ""))
                if self.task["kind"] != "place" or row is None or action[7] != row["slot"]:
                    raise ValueError("placement does not match task inventory slot")
            if self.task["kind"] == "gather" and self.task["item"] in {"iron_ore", "coal", "cobblestone"}:
                tool = self._tool(state)
                if action[5] == 3 and (tool is None or not tool.get("equipped")):
                    raise ValueError("mining requires equipped pickaxe")
            if self.collect and action[5] == 3:
                action[5] = 0
                self.feedback = "collection active; attack removed, inspect drops"
            if (self.task["kind"] == "place"
                    and self._blocked("place:" + self.task["item"], state) and action[5] == 6):
                raise ValueError("placement cooling down; reposition first")
            duration = min(steps, self.settings["max_action_steps"])
            moving = bool(action[0] or action[1])
            rotating = action[3] != 12 or action[4] != 12
            if rotating:
                self._invalidate_path(state, "camera_changed")
            if (moving and not rotating and image and not self._move_hazard(state)
                    and decision.get("safe_path") is True):
                self.visual_move = {"position": state["pos"],
                                    "yaw": state["orientation"].get("yaw", 0),
                                    "motion": [action[0], action[1]],
                                    "remaining": self.settings["visual_move_steps"],
                                    "until": state["step"] + self.settings["visual_move_steps"]}
                self._log("path_created", step=state["step"], motion=self.visual_move["motion"])
                duration = min(duration, self.settings["visual_move_steps"])
            if self.collect or self.target or self.task["kind"] in {"place", "return"}:
                duration = min(duration, 4)
            if moving and (rotating or not self._safe_move(state)
                           or self.visual_move["motion"] != action[:2]):
                action[0] = action[1] = 0
                duration = 1
                self.feedback = "movement withheld; correct view or confirm safe path"
                self.metrics["movement_stripped"] += 1
                self._log("action_stripped", step=state["step"], component="movement", reason=self.feedback)
            self.segment = {"action": action, "remaining": duration,
                            "first": True, "task": dict(self.task)}
            if (self.task["kind"] == "gather" and self.task["item"] == "log"
                    and not self.target and not self.collect and self.route is None
                    and (action[0] or action[1]) and action[5] == 0):
                self.route = {"origin": state["pos"], "start": state["step"],
                              "yaw": state["orientation"].get("yaw", 0),
                              "action": list(action), "verified": False}
            self.model_errors = 0
            self.decision_errors = 0
            self.metrics["attack_proposed"] += int(proposed_attack)
            if self.handoff:
                self._log("model_control_started", step=state["step"],
                          waited=state["step"] - self.handoff["start"])
                self.handoff = None
                self.model_control_until = state["step"] + self.settings["model_control_steps"]
                self.recovery_hold_until = self.model_control_until
            self.metrics["model_success"] += 1
            self._log("decision", step=state["step"], task=self.task,
                      action=action, steps=self.segment["remaining"],
                      visual_path=bool(self.visual_move),
                      model_control=state["step"] < self.model_control_until)
            return True
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
            if stage == "request":
                self.model_errors += 1
                factor = min(self.model_errors, self.settings["model_error_limit"])
                self.next_model_step = state["step"] + self.settings["model_backoff_steps"] * factor
                self.next_model_time = time.monotonic() + self.settings["model_backoff_seconds"] * factor
            else:
                self.model_errors = 0
                self.decision_errors += 1
                self.next_model_step = state["step"] + self.settings["model_min_interval"]
                self.next_model_time = 0
            if self.handoff:
                self.handoff["status"] = "waiting_request"
            self.metrics["model_" + stage + "_error"] += 1
            code = error.code if isinstance(error, ModelResponseError) else type(error).__name__
            detail = str(error)[:160] if stage == "validation" else None
            self.feedback = "model " + stage + " failed: " + code
            if detail:
                self.feedback += ": " + detail
            response = self.client.last_response or {}
            upstream_code = response.get("error_code")
            if upstream_code in {"token_limit", "accounting_error"}:
                self.terminal_model_error = True
            self._log("model_error", step=state["step"], stage=stage, code=code, upstream_code=upstream_code,
                      detail=detail, http_status=response.get("http_status"),
                      finish_reason=response.get("finish_reason"),
                      request_id=str(response.get("request_id") or "")[:128],
                      elapsed_seconds=round(time.monotonic() - request_started, 2),
                      consecutive=self.model_errors)
            if stage == "request" and self.model_errors >= self.settings["model_error_limit"]:
                self._handoff(state, "model response unavailable; wait for scheduled retry without repeating recovery")
            return False
        finally:
            response = self.client.last_response or {}
            usage = response.get("usage", {})
            self.tokens += usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

    def _segment_action(self, state, obs):
        segment = self.segment
        if not segment or segment["remaining"] <= 0 or segment["task"] != self.task:
            self.segment = None
            return None
        action = list(segment["action"])
        if action[5] == 3:
            target_key = (state["block"], *(state["hit"].get("traced_block_" + a) for a in "xyz"))
            hit_distance = state["hit"].get("block_distance")
            reachable = isinstance(hit_distance, (int, float)) and 0 <= hit_distance <= self.settings["interaction_distance"]
            if segment["first"] and (action[3] != 12 or action[4] != 12):
                action[5] = 0
                self.alignment_pending = True
                self._phase("align", state)
                self._log("attack_align", step=state["step"])
            elif (self.collect or self.task["kind"] != "gather" or not reachable
                  or not resource_matches(self.task["item"], state["block"])
                  or state["step"] < self.mining_cooldown.get(target_key, 0)
                  or (self.task["item"] == "log" and self.failed_trees.get(target_key[1:], 0) > state["step"])):
                action[5] = 0
                segment["action"][5] = 0
                self.metrics["attack_rejected_target"] += 1
                self.feedback = "attack removed: target, reach, collection or cooldown"
                self._log("attack_rejected", step=state["step"], reason=self.feedback,
                          block=state["block"], distance=hit_distance)
        if (action[0] or action[1]) and (not self._safe_move(state)
                or self.visual_move["motion"] != action[:2]):
            action[0] = action[1] = 0
            segment["action"][0] = segment["action"][1] = 0
            self.feedback = "movement removed; path permission expired"
        if not segment["first"]:
            action[3] = action[4] = 12
            if action[5] in (1, 6):
                action[5] = 0
        segment["first"] = False
        segment["remaining"] -= 1
        if action[5] == 3:
            key = (state["block"], *(state["hit"].get("traced_block_" + a) for a in "xyz"))
            if key != self.attack_key:
                self.attack_key, self.attack_steps = key, 0
            self.attack_steps += 1
            if self.attack_steps > self.settings["attack_timeout"]:
                self.mining_cooldown[key] = state["step"] + self.settings["mining_cooldown"]
                self.attack_key = None
                self.attack_steps = 0
                self.segment = None
                self._handoff(state, "model mining timed out; check reach and target")
                return list(obs["noop_action"])
            action[0] = action[1] = action[2] = 0
            self._phase("mine", state)
        if action[5] == 6:
            return self._issue(action, "place", self.task["item"], state)
        return action

    def _level_view(self, state, obs):
        pitch = state["orientation"].get("pitch", 0)
        if abs(pitch) <= self.settings["level_pitch"]:
            return None
        action = list(obs["noop_action"])
        action[3] = 13 if pitch < 0 else 11
        self._phase("align", state)
        return action

    def _finish_collect(self, reason, state):
        if self.collect:
            self._log("collect_finished", step=state["step"], reason=reason,
                      elapsed=state["step"] - self.collect["start"],
                      resource=self.collect["item"])
            self.metrics["collect_" + reason] += 1
        self.collect = None
        self.search_origin = state["pos"]
        self.search_started = state["step"]
        self.segment = None
        self._phase("search", state)

    def _move_hazard(self, state):
        if (state["health"] <= self.settings["low_health"]
                or abs(state["orientation"].get("pitch", 0)) > self.settings["max_move_pitch"]
                or state["block"] in {"water", "flowing_water", "lava", "flowing_lava"}):
            return True
        return bool(self.previous and (
            state["health"] < self.previous["health"]
            or state["pos"][1] < self.previous["pos"][1] - 0.6))

    def _safe_move(self, state):
        if self._move_hazard(state):
            return False
        permit = self.visual_move
        return bool(permit and permit["remaining"] > 0 and state["step"] <= permit["until"]
                    and abs((state["orientation"].get("yaw", 0) - permit["yaw"] + 180) % 360 - 180) <= 7.5
                    and abs(state["pos"][1] - permit["position"][1]) < 1.25)

    def _navigate(self, target, state, obs, arrival=1.0, block_center=True,
                  horizontal_only=False):
        if not all(isinstance(v, (int, float)) for v in target):
            return None
        offset = 0.5 if block_center else 0.0
        dx, dz = target[0] + offset - state["pos"][0], target[2] + offset - state["pos"][2]
        if math.hypot(dx, dz) <= arrival:
            return None
        action = list(obs["noop_action"])
        desired = math.degrees(math.atan2(-dx, dz))
        error = (desired - state["orientation"].get("yaw", 0) + 180) % 360 - 180
        if abs(error) > 12:
            action[4] = 13 if error > 0 else 11
            return action
        if (self._safe_move(state) and self.visual_move["motion"] == [1, 0]
                and (horizontal_only or abs(target[1] - state["pos"][1]) < 2.5)):
            action[0], action[2] = 1, 2
            hit_distance = state["hit"].get("block_distance")
            if (self.task["kind"] == "gather" and self.task["item"] == "log"
                    and not self.collect and resource_matches("log", state["block"])
                    and isinstance(hit_distance, (int, float))
                    and hit_distance > self.settings["interaction_distance"] + 1):
                action[2] = 0
            return action
        return None

    def _collect_action(self, state, obs):
        if not self.collect:
            return None
        self._phase("collect", state)
        target = self.collect["position"]
        elapsed = state["step"] - self.collect["start"]
        if elapsed >= self.settings["collect_steps"]:
            self.feedback = "drop unconfirmed after bounded collection; recheck tree"
            self._finish_collect("timeout", state)
            return None
        if all(isinstance(v, (int, float)) for v in target):
            horizontal = math.hypot(target[0] + 0.5 - state["pos"][0],
                                    target[2] + 0.5 - state["pos"][2])
            if horizontal < 1.2:
                if self.collect["arrived"] is None:
                    self.collect["arrived"] = state["step"]
                    self._log("collect_arrived", step=state["step"], position=state["pos"])
                if state["step"] - self.collect["arrived"] >= self.settings["collect_arrival_steps"]:
                    self._finish_collect("arrival_unconfirmed", state)
                    return None
            else:
                self.collect["arrived"] = None
            action = self._navigate(target, state, obs, arrival=0.6,
                                    horizontal_only=True)
            if action is not None:
                return action
        self.feedback = "collection path uncertain; inspect saved drop position, do not search a new tree"
        return None

    def _wood_navigation(self, state, obs):
        if self.task["kind"] != "gather" or self.task["item"] != "log" or self.collect:
            return None
        hit = state["hit"]
        self.failed_trees = {pos: until for pos, until in self.failed_trees.items()
                             if until > state["step"]}
        if resource_matches("log", state["block"]):
            position = tuple(hit.get("traced_block_" + a) for a in "xyz")
            if position in self.failed_trees:
                self.target = self.segment = self.route = None
                self._phase("search", state)
                self.feedback = "failed tree cooling down; turn toward another search region"
                action = list(obs["noop_action"])
                action[4] = 14 if self.recovery % 2 else 10
                return action
            if (all(isinstance(v, (int, float)) for v in position)
                    and position not in self.failed_trees and not self.target):
                self.target = {"position": position, "start": state["step"],
                               "best_distance": distance(position, state["pos"]),
                               "progress_step": state["step"]}
                self.segment = None
                self.last_navigation_review = state["step"]
                self._log("tree_target", step=state["step"], position=position)
        if self.target:
            current_distance = distance(self.target["position"], state["pos"])
            if current_distance < self.target["best_distance"] - 0.25:
                self.target["best_distance"] = current_distance
                self.target["progress_step"] = state["step"]
            if (state["step"] - self.target["start"] >= self.settings["target_lifetime"]
                    or state["step"] - self.target["progress_step"] >= self.settings["navigation_timeout"]):
                position = self.target["position"]
                self.memory.append({"failed_tree": position, "step": state["step"]})
                self.failed_trees[position] = state["step"] + self.settings["target_lifetime"]
                self.target = self.segment = None
                self.search_origin = state["pos"]
                self.search_started = state["step"]
                self._handoff(state, "tree approach stalled; choose alternate access or another visible trunk")
                return None
            if self._model_controls(state):
                return None
            if state["step"] - self.last_navigation_review >= self.settings["navigation_review_steps"]:
                self.last_navigation_review = state["step"] + self.settings["model_control_steps"]
                dx = self.target["position"][0] + 0.5 - state["pos"][0]
                dz = self.target["position"][2] + 0.5 - state["pos"][2]
                yaw_error = (math.degrees(math.atan2(-dx, dz))
                             - state["orientation"].get("yaw", 0) + 180) % 360 - 180
                self._log("tree_approach", step=state["step"], distance=round(current_distance, 2),
                          yaw_error=round(yaw_error, 1),
                          no_progress=state["step"] - self.target["progress_step"])
                self._handoff(state, "review tree approach using image; choose safe speed or an alternate path")
                return None
            action = self._navigate(self.target["position"], state, obs, arrival=1.8)
            if action is not None:
                self._phase("approach", state)
                return action
            self._phase("align", state)
        if self.route and state["step"] - self.route["start"] >= self.settings["route_steps"]:
            self.memory.append({"explored_from": self.route["origin"], "yaw": self.route["yaw"],
                                "position": state["pos"], "step": state["step"]})
            self.route = self.segment = None
            self.feedback = "search leg finished; choose a new region or visible tree"
        return None

    def _fallback(self, state, obs):
        leveled = self._level_view(state, obs)
        if leveled is not None:
            return leveled
        searching = (self.task["kind"] == "gather" and self.task["item"] == "log"
                     and not self.collect and not self.target)
        if searching and self.route and self.last_action and (self.last_action[0] or self.last_action[1]):
            self.route["verified"] = self._safe_move(state) and self.previous is not None and distance(state["pos"], self.previous["pos"]) > 0.03
            if self.route["verified"]:
                self.route["action"] = list(self.last_action)
        action = list(obs["noop_action"])
        if searching and self.route and self.route["verified"] and self._safe_move(state):
            action = list(self.route["action"])
            action[3] = action[4] = 12
            action[5] = 0
            self.metrics["route_continued"] += 1
        elif state["step"] % 6 == 0:
            action[4] = 13 if self.recovery % 2 == 0 else 11
        self.metrics["fallback_steps"] += 1
        return action

    def _summary(self, state):
        if state["step"] - self.last_summary < self.settings["progress_interval"]:
            return
        self.last_summary = state["step"]
        self._log("progress", step=state["step"], task=self.task, phase=self.phase,
                  counts=dict(state["counts"]), material_requirements=self.wood_requirements,
                  no_resource_progress=state["step"] - self.last_resource_step,
                  task_age=state["step"] - self.task_started_step,
                  phase_age=state["step"] - self.phase_started_step,
                  no_navigation_progress=state["step"] - self.last_navigation_step,
                  metrics=dict(self.metrics), recoveries=self.recovery,
                  planning=self.handoff,
                  planning_wait_steps=(state["step"] - self.handoff["start"] if self.handoff else 0),
                  retry_in_steps=max(0, self.next_model_step - state["step"]),
                  retry_in_seconds=round(max(0, self.next_model_time - time.monotonic()), 2),
                  elapsed_seconds=round(time.monotonic() - self.started, 1), tokens=self.tokens)

    def act(self, observation):
        self.observation = observation
        state = self._observe(observation)
        self._verify(state, observation)
        task = self._plan(state)
        if task != self.task:
            self.segment = None
            self.task = task
            self.attack_steps = 0
            if task["kind"] != "gather":
                self.target = self.route = None
            if not self.collect or task["kind"] != "gather" or task["item"] != self.collect["item"]:
                self.collect = None
            self.task_started_step = state["step"]
            if task["kind"] == "gather" and task["item"] == "log":
                self.search_origin = state["pos"]
                self.search_started = state["step"]
                self.target = self.route = None
            self._phase("plan", state)
            self._log("task", step=state["step"], task=task)
        if self.search_origin is None:
            self.search_origin = state["pos"]
            self.search_started = state["step"]
        wood_search = (self.task["kind"] == "gather" and self.task["item"] == "log"
                       and not self.collect)
        if (wood_search and not self.target and not self._model_controls(state)
                and state["step"] - self.search_started >= self.settings["search_timeout"]):
            continuing = (self.segment and self.segment["remaining"] > 0
                          and (self.segment["action"][0] or self.segment["action"][1])
                          and self._safe_move(state) and not self._stuck(state))
            if continuing:
                if not self.search_replan:
                    self._log("search_replan_deferred", step=state["step"])
                self.search_replan = True
            else:
                displacement = distance(self.search_origin, state["pos"])
                self.memory.append({"search_origin": self.search_origin, "end": state["pos"],
                                    "displacement": round(displacement, 2), "step": state["step"]})
                self.search_origin = state["pos"]
                self.search_started = state["step"]
                self.search_replan = False
                self.route = None
                self._handoff(state, "search interval found no log; choose a new visible trunk or search region")
        action = None
        if self.pending:
            action = list(observation["noop_action"])
        elif state["health"] <= 0:
            action = list(observation["noop_action"])
        elif (not self._model_controls(state) and state["step"] >= self.recovery_hold_until
              and (self.recovery_requested or self.recovery_remaining or self._stuck(state))):
            self.segment = None
            self.feedback = "movement stuck; try another route"
            action = self._recover(state, observation)
        else:
            if state["health"] <= self.settings["low_health"]:
                self.feedback = "low health; avoid hazards while continuing the task"
            hit_distance = state["hit"].get("block_distance")
            can_mine = (self.task["kind"] == "gather" and not self.collect
                        and resource_matches(self.task["item"], state["block"])
                        and isinstance(hit_distance, (int, float))
                        and 0 <= hit_distance <= self.settings["interaction_distance"])
            action = self._level_view(state, observation) if not can_mine else None
            if action is None:
                action = self._collect_action(state, observation)
            if (action is None and not self.collect
                    and (self.task["kind"] != "gather" or can_mine
                         or self.alignment_pending or not self._model_controls(state))):
                action = self._execute(state, observation)
            self.alignment_pending = False
            if (action is None and not self.collect
                    and self.task["kind"] == "gather"
                    and self.task["item"] == "log"):
                action = self._wood_navigation(state, observation)
            if action is not None:
                self.segment = None
            if action is None:
                action = self._segment_action(state, observation)
            if action is None and self._ask_model(state, observation):
                action = self._segment_action(state, observation)
            if action is None and not self._model_controls(state):
                action = self._fallback(state, observation)
            elif action is None:
                action = self._level_view(state, observation) or list(observation["noop_action"])
        if not valid_action(action):
            action = list(observation["noop_action"])
            self.feedback = "invalid internal action discarded"
        if action[3] != 12 or action[4] != 12:
            self._invalidate_path(state, "camera_action")
        if action[0] or action[1]:
            if (not self._safe_move(state) or self.visual_move["motion"] != action[:2]):
                action[0] = action[1] = 0
                self.metrics["movement_stripped"] += 1
            else:
                self.visual_move["remaining"] -= 1
        if self.segment and self.segment["remaining"] == 0:
            self._log("segment_finished", step=state["step"])
            self.segment = None
        if action[5] == 3:
            self.metrics["attack_emitted"] += 1
        self._summary(state)
        self.previous = state
        self.last_action = action
        self.node = "observe"
        return action
