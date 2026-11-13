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
    "progress_interval": 200, "search_timeout": 160,
    "collect_steps": 16, "budget_reserve_seconds": 5,
    "token_input_reserve": 16000, "mining_cooldown": 30,
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
        self.last_progress_step = 0
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
                self.facilities[facility] = {"position": state["pos"], "step": state["step"]}
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
        if previous:
            moved = distance(state["pos"], previous["pos"])
            changed = state["counts"] != previous["counts"]
            if changed:
                self.segment = None
                self.attack_steps = 0
                self.collect = None
                self.attack_key = None
                self.last_progress_step = state["step"]
            if (not changed and self.last_action and self.last_action[5] == 3
                    and self.task and self.task["kind"] == "gather" and self.attack_key
                    and resource_matches(self.task["item"], previous["block"])):
                old_pos = self.attack_key[1:]
                new_pos = tuple(state["hit"].get("traced_block_" + a) for a in "xyz")
                if (not resource_matches(self.task["item"], state["block"])
                        or (all(v is not None for v in old_pos + new_pos) and old_pos != new_pos)):
                    self.collect = {"position": old_pos, "start": state["step"],
                                    "item": self.task["item"]}
                    self.phase = "collect"
                    self.attack_key = None
                    self.segment = None
                    self.feedback = "target changed after mining; check nearby drops"
            if state["health"] < previous["health"]:
                self.segment = None
                self.feedback = "health decreased; avoid hazards"
            lost = sum(state["counts"].values()) < sum(previous["counts"].values())
            if moved > 12 or (lost and previous["health"] <= 0 < state["health"]):
                self.pending = self.segment = None
                self.collect = None
                self.attack_key = None
                self.recent.clear()
                self.feedback = "possible respawn or displacement; recheck resources"
                self._log("state_discontinuity", step=state["step"])
        last = obs.get("last_action") or {}
        if last.get("valid") is False:
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

    def _plan(self, state):
        self.node = "plan"
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
                self.phase = "mine"
                self.metrics["attack_rule"] += 1
                return action
            self.feedback = "mining timeout; check tool, reach and target"
            self.mining_cooldown[key] = state["step"] + self.settings["mining_cooldown"]
            self.attack_key = None
            self.attack_steps = 0
            self.segment = None
            self.recovery_requested = True
            self._log("mining_timeout", step=state["step"], block=state["block"], distance=hit_distance)
        elif kind == "gather":
            self.phase = "approach" if resource_matches(name, state["block"]) else "search"
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
        leveled = self._level_view(state, obs)
        if leveled is not None:
            return leveled
        self.recovery_requested = False
        action = list(obs["noop_action"])
        if not self.recovery_remaining:
            self.recovery += 1
            self.recovery_remaining = 4
            self.memory.append({"failed_position": state["pos"], "step": state["step"]})
            self.recent.clear()
        hit_distance = state["hit"].get("block_distance")
        if (isinstance(hit_distance, (int, float)) and hit_distance < 3
                and state["block"] not in {"water", "flowing_water", "lava", "flowing_lava"}
                and state["health"] > self.settings["low_health"]):
            action[1] = 1 if self.recovery % 2 else 2
            action[2] = 1
        if self.recovery_remaining == 4:
            action[4] = 14 if self.recovery % 2 else 10
        self.recovery_remaining -= 1
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
            return False
        if state["step"] < self.next_model_step or now < self.next_model_time:
            return False
        self.client.timeout = min(self.settings["model_timeout"], remaining - self.settings["budget_reserve_seconds"])
        self.client.max_output_tokens = min(self.settings["max_output_tokens"], self.settings["model_token_budget"] - self.tokens)
        prompt = (
            "Control Minecraft through MineDojo. Finish the current task quickly and reliably. "
            "Return ONLY JSON: {\"action\":[8 integers],\"steps\":integer,\"reason\":\"short\"}. "
            "Action dimensions: forward 0=still/1=forward/2=back; strafe 0=still/1=left/2=right; "
            "stance 0=normal/1=jump/2=sneak/3=sprint; pitch and yaw 0..24,12=no rotation; "
            "operation 0=noop/1=use/2=drop/3=attack/4=craft-smelt/5=equip/6=place/7=destroy; "
            "craft index 0..243; inventory slot 0..35. Start from the supplied noop. "
            "Only operations 0,1,3 are allowed, except operation 6 when the task is place. "
            "Crafting and equipping are handled by the controller. Do not drop/destroy items. "
            "For place use the actual inventory slot of the requested facility, aim at reachable "
            "solid ground with free space, avoid placing inside yourself; it executes once. "
            "For failed craft, reposition near its required facility or diagnose missing fuel. "
            "For return, navigate toward the remembered facility and verify it; no teleport. "
            "Ray block wood is a tree log; inventory may call it log or log2. "
            "Use phase and collection.position: safely collect drops before searching again. "
            "Find trees for log, exposed stone for cobblestone, coal ore for coal, iron ore for iron. "
            "Use the image for directions; one crosshair ray is not a map or proof of reach. "
            "Avoid water, lava, cliffs, digging straight down and breaking our facilities. "
            "Mining requires suitable equipped pickaxes; keep a stable aim. "
            "When a block breaks, move safely toward drops if inventory has not increased. "
            "Rotation executes once, movement/attack can repeat for steps, up to "
            + str(self.settings["max_action_steps"]) + ". Prefer short alignment actions. "
            "Do not assume angular scale; compare subsequent orientation feedback. "
            "Search new ground instead of repeating failed actions. State:\n"
            + json.dumps({"noop": obs["noop_action"], **self._context(state, obs)}, ensure_ascii=False)
        )
        self.calls += 1
        self.next_model_step = state["step"] + self.settings["model_min_interval"]
        stage = "request"
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
            self.segment = {"action": action, "remaining": min(steps, self.settings["max_action_steps"]),
                            "first": True, "task": dict(self.task)}
            self.model_errors = 0
            self.metrics["model_success"] += 1
            self.metrics["attack_proposed"] += int(action[5] == 3)
            self._log("decision", step=state["step"], task=self.task,
                      action=action, steps=self.segment["remaining"])
            return True
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
            self.model_errors += 1
            factor = min(self.model_errors, self.settings["model_error_limit"])
            self.next_model_step = state["step"] + self.settings["model_backoff_steps"] * factor
            self.next_model_time = time.monotonic() + self.settings["model_backoff_seconds"] * factor
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
                      consecutive=self.model_errors)
            if self.model_errors >= self.settings["model_error_limit"]:
                self.recovery_requested = True
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
            if self.collect:
                self.segment = None
                self.feedback = "collect the previous drop before attacking"
                return None
            target_key = (state["block"], *(state["hit"].get("traced_block_" + a) for a in "xyz"))
            if state["step"] < self.mining_cooldown.get(target_key, 0):
                self.segment = None
                self.feedback = "mining target cooling down; reposition"
                return None
            hit_distance = state["hit"].get("block_distance")
            reachable = isinstance(hit_distance, (int, float)) and 0 <= hit_distance <= self.settings["interaction_distance"]
            if segment["first"] and (action[3] != 12 or action[4] != 12):
                action[5] = 0
                self.phase = "align"
            elif (self.task["kind"] != "gather" or not reachable
                  or not resource_matches(self.task["item"], state["block"])):
                if segment["first"] and (action[3] != 12 or action[4] != 12):
                    action[5] = 0
                else:
                    self.segment = None
                    self.metrics["attack_rejected_target"] += 1
                    self.feedback = "attack stopped: target mismatch or outside reach"
                    self._log("attack_rejected", step=state["step"], reason="target_or_reach", block=state["block"], distance=hit_distance)
                    return None
        if not segment["first"]:
            action[3] = action[4] = 12
            if action[5] in (1, 6):
                action[5] = 0
        segment["first"] = False
        segment["remaining"] -= 1
        if action[5] == 6:
            return self._issue(action, "place", self.task["item"], state)
        return action

    def _level_view(self, state, obs):
        pitch = state["orientation"].get("pitch", 0)
        if abs(pitch) <= 65:
            return None
        action = list(obs["noop_action"])
        action[3] = 13 if pitch < 0 else 11
        self.phase = "align"
        return action

    def _collect_action(self, state, obs):
        if not self.collect:
            return None
        if state["step"] - self.collect["start"] >= self.settings["collect_steps"]:
            self.collect = None
            self.feedback = "collection unconfirmed; reobserve drops"
            return None
        self.phase = "collect"
        action = list(obs["noop_action"])
        target = self.collect["position"]
        hit = state["hit"].get("block_distance")
        if not isinstance(hit, (int, float)) or hit >= 4 or not all(v is not None for v in target):
            return None
        if (all(v is not None for v in target) and state["health"] > self.settings["low_health"]
                and abs(target[1] - state["pos"][1]) < 1.5
                and state["block"] not in {"water", "flowing_water", "lava", "flowing_lava"}
                and isinstance(hit, (int, float)) and hit < 4):
            dx, dz = target[0] + 0.5 - state["pos"][0], target[2] + 0.5 - state["pos"][2]
            desired = math.degrees(math.atan2(-dx, dz))
            error = (desired - state["orientation"].get("yaw", 0) + 180) % 360 - 180
            if abs(error) > 15:
                action[4] = 13 if error > 0 else 11
            elif math.hypot(dx, dz) > 0.6:
                action[0], action[2] = 1, 2
        return action if action != list(obs["noop_action"]) else None

    def _fallback(self, state, obs):
        leveled = self._level_view(state, obs)
        if leveled is not None:
            return leveled
        action = list(obs["noop_action"])
        if state["step"] % 6 == 0:
            action[4] = 13 if self.recovery % 2 == 0 else 11
        self.metrics["fallback_steps"] += 1
        return action

    def _summary(self, state):
        if state["step"] - self.last_summary < self.settings["progress_interval"]:
            return
        self.last_summary = state["step"]
        self._log("progress", step=state["step"], task=self.task, phase=self.phase,
                  counts=dict(state["counts"]), no_resource_progress=state["step"] - self.last_progress_step,
                  metrics=dict(self.metrics), recoveries=self.recovery,
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
            self.collect = None
            self.last_progress_step = state["step"]
            self._log("task", step=state["step"], task=task)
        if (self.task["kind"] == "gather" and not self.collect
                and state["step"] - self.last_progress_step >= self.settings["search_timeout"]
                and self.phase != "mine"):
            self.feedback = "search made no progress; change view or route"
            self.recovery_requested = True
            self.last_progress_step = state["step"]
        action = None
        if self.pending:
            action = list(observation["noop_action"])
        elif state["health"] <= 0:
            action = list(observation["noop_action"])
        elif self.recovery_requested or self.recovery_remaining or self._stuck(state):
            self.segment = None
            self.feedback = "movement stuck; try another route"
            action = self._recover(state, observation)
        else:
            if state["health"] <= self.settings["low_health"]:
                self.feedback = "low health; avoid hazards while continuing the task"
            action = self._collect_action(state, observation)
            if action is None and not self.collect:
                action = self._execute(state, observation)
            if action is None:
                action = self._segment_action(state, observation)
            if action is None and self._ask_model(state, observation):
                action = self._segment_action(state, observation)
            if action is None:
                action = self._fallback(state, observation)
        if not valid_action(action):
            action = list(observation["noop_action"])
            self.feedback = "invalid internal action discarded"
        if action[5] == 3:
            self.metrics["attack_emitted"] += 1
        self._summary(state)
        self.previous = state
        self.last_action = action
        self.node = "observe"
        return action
