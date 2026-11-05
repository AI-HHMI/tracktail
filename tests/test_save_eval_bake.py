#!/usr/bin/env python
"""Change C control-flow test: save_checkpoint bakes model_state_eval for muon_schedulefree
(not just scheduler_type=='schedulefree'), the eval/train snapshots differ, the optimizer is
restored to train mode, and non-schedule-free runs do NOT bake. Uses a fake schedule-free
optimizer (eval = train + 0.5) so no VJEPA / Muon / distributed setup is needed."""
import os
import tempfile

import torch
import torch.nn as nn

from tracktail.model.train_utils import save_checkpoint


class C(dict):
    __getattr__ = dict.get  # attribute access + .get, like the real config object


class FakeSF:
    """Mimics a schedule-free optimizer's in-place train(y)/eval(x) parameter swap."""
    def __init__(self, model):
        self.model = model
        self._mode = 'train'

    def state_dict(self):
        return {'fake': 1}

    def train(self):
        if self._mode == 'eval':
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(-0.5)
            self._mode = 'train'

    def eval(self):
        if self._mode == 'train':
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(0.5)
            self._mode = 'eval'


def _load(d, i):
    return torch.load(os.path.join(d, 'checkpoints', f'checkpoint_{str(i).zfill(8)}.pth'),
                      weights_only=False)


def main():
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    opt = FakeSF(model)
    ok = True

    for muon_sf, sched in [(True, 'muon'), (False, 'schedulefree')]:
        cfg = C(training=C(scheduler_type=sched, optimizer=C(muon_schedulefree=muon_sf)))
        d = tempfile.mkdtemp()
        save_checkpoint(model, opt, prefix=d, i=10, config=cfg)
        ckpt = _load(d, 10)
        has_eval = 'model_state_eval' in ckpt
        delta = ckpt['model_state_eval']['weight'] - ckpt['model_state']['weight']
        delta_ok = torch.allclose(delta, torch.full_like(delta, 0.5))
        restored = (opt._mode == 'train')
        good = has_eval and delta_ok and restored
        ok &= good
        print(f'{"PASS" if good else "FAIL"} muon_sf={muon_sf} sched={sched}: '
              f'baked={has_eval} delta=+0.5={delta_ok} restored_train={restored}')

    # negative: a non-schedule-free run must NOT bake model_state_eval
    cfg = C(training=C(scheduler_type='constant', optimizer=C()))
    d = tempfile.mkdtemp()
    save_checkpoint(model, opt, prefix=d, i=20, config=cfg)
    ckpt = _load(d, 20)
    neg_ok = 'model_state_eval' not in ckpt
    ok &= neg_ok
    print(f'{"PASS" if neg_ok else "FAIL"} non-SF (constant): no model_state_eval')

    print('\n=== SAVE EVAL-BAKE:', 'PASS ===' if ok else 'FAIL ===')
    import sys
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
