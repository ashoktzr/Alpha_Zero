# app/progress.py
"""
ProgressEmitter: simplified event emitter for UI.

Behavior:
- Do NOT emit percentages by default.
- Emit stage_started(stage_name)
- Emit stage_finished(stage_name)
- Optional: emit subtask messages.
- Clients (GUI) should display a spinner + a checklist of completed stages.
"""

import time
from typing import Callable

class ProgressEmitter:
    def __init__(self, callback: Callable[[dict], None]=None):
        """
        callback: function taking one dict event per update
        Event schema examples:
          {'event': 'start', 'stage': 'indicators'}
          {'event': 'info', 'stage': 'indicators', 'msg': 'computed macd(12,26,9)'}
          {'event': 'done', 'stage': 'indicators'}
        """
        self.callback = callback or (lambda ev: None)
        self._stages_done = []

    def emit_start(self, stage, msg=None):
        ev = {'event':'start', 'stage': stage, 'msg': msg}
        self.callback(ev)

    def emit_info(self, stage, msg):
        ev = {'event':'info', 'stage': stage, 'msg': msg}
        self.callback(ev)

    def emit_done(self, stage, msg=None):
        if stage not in self._stages_done:
            self._stages_done.append(stage)
        ev = {'event':'done', 'stage': stage, 'msg': msg}
        self.callback(ev)

    def stages_done(self):
        return list(self._stages_done)
