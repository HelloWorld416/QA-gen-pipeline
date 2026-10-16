"""Persistent attempt ceiling and optional provider-reported token stop threshold."""
import threading
from pathlib import Path
from .common import dump_json, load_json


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, path, max_attempts, max_reported_tokens=0):
        self.path=Path(path);self.lock=threading.Lock()
        self.max_attempts=max_attempts;self.max_reported_tokens=max_reported_tokens
        self.state=load_json(self.path) if self.path.exists() else dict(attempts_reserved=0,reported_tokens=0,unsettled_attempts=0)

    def reserve(self):
        with self.lock:
            if self.state['attempts_reserved']>=self.max_attempts:
                raise BudgetExhausted('Persistent API attempt ceiling reached')
            if self.max_reported_tokens and self.state['reported_tokens']>=self.max_reported_tokens:
                raise BudgetExhausted('Provider-reported token stop threshold reached')
            self.state['attempts_reserved']+=1;self.state['unsettled_attempts']+=1
            dump_json(self.state,self.path)

    def finish(self, usage=None):
        with self.lock:
            usage=usage or {}
            self.state['reported_tokens']+=usage.get('total_tokens',usage.get('input_tokens',0)+usage.get('output_tokens',0))
            self.state['unsettled_attempts']-=1
            dump_json(self.state,self.path)
