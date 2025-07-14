#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.optim import lr_scheduler

class Scheduler():
    def __str__(self):
        sched_str = '%s (\n' % self.name
        for key in vars(self).keys():
            if key != 'name':
                value = vars(self)[key]
                if key == 'optimizer': value = str(value).replace('\n', '\n        ').replace('    )', ')')
                sched_str +=  '    %s: %s\n' % (key, value)
        sched_str += ')'
        return sched_str

class fixed(Scheduler):
    def __init__(self, optimizer, sched_args):
        self.name = 'FixedLR'
        self.optimizer = optimizer

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        pass

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

class step(lr_scheduler.StepLR, Scheduler):
   def __init__(self, optimizer, sched_args):
       super(step, self).__init__(optimizer, **sched_args)
       self.name = 'StepLR'

class const(lr_scheduler.ConstantLR, Scheduler):
    def __init__(self, optimizer, sched_args):
       super(const, self).__init__(optimizer, **sched_args)
       self.name = 'ConstantLR'

class plateau_loss(lr_scheduler.ReduceLROnPlateau, Scheduler):
    def __init__(self, optimizer, sched_args):
       super(plateau_loss, self).__init__(optimizer, **sched_args)
       self.name = 'ReduceLROnPlateauLoss'
