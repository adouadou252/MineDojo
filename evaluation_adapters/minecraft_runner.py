"""Compatibility imports. The active controller is pdf_runner, not the old split-goal runner.

Use practice.py for fresh per-world processes and independent wall/token budgets.
Do not run several seeds through one persistent Agent connection.
"""
from .pdf_runner import (checked_action, create_minedojo_environment, public_observation,
                         run_episode, validate_scenarios)
