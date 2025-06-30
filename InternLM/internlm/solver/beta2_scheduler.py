#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import numpy as np


class Beta2Scheduler:
    """
    Beta2Scheduler
    """

    def __init__(self, optimizer: torch.optim.Adam, init_beta2, c=0.8, cur_iter=-1, schedule_wd=False, total_iter=21000, init_wd=0.04, final_wd=0.4):
        self.cur_iter = 0 if cur_iter == -1 else cur_iter
        self.init_beta2 = init_beta2
        self.c = c
        self.optimizer = optimizer

        self.schedule_wd = schedule_wd
        self.init_wd = init_wd
        self.final_wd = final_wd
        self.total_iter = total_iter

        assert isinstance(
            optimizer, (torch.optim.Adam, torch.optim.AdamW)
        ), "should use Adam optimzier, which has beta2"

    def step(self, cur_iter=None):
        if cur_iter is None:
            self.cur_iter += 1
        else:
            self.cur_iter = cur_iter

        if not self.schedule_wd:
            new_beta2 = self.get_beta2()
            for pg in self.optimizer.param_groups:
                beta1, _ = pg["betas"]
                pg["betas"] = (beta1, new_beta2)
        else:
            new_beta2 = self.get_beta2()
            new_wd = self.get_wd()
            for pg in self.optimizer.param_groups:
                beta1, _ = pg["betas"]
                pg["betas"] = (beta1, new_beta2)
                pg["weight_decay"] = new_wd

    def get_beta2(self):
        if self.c <= 0:
            return self.init_beta2
        scale = 1 - (1 / self.cur_iter**self.c)
        return max(self.init_beta2, scale)

    def get_wd(self):
        if not self.schedule_wd:
            return self.init_wd
        wd = self.final_wd + 0.5 * (self.init_wd - self.final_wd) * (1 + np.cos(np.pi * self.cur_iter / self.total_iter))
        return wd


# =======================================
# Original Beta2Scheduler
# =======================================

# class Beta2Scheduler:
#     """
#     Beta2Scheduler
#     """

#     def __init__(self, optimizer: torch.optim.Adam, init_beta2, c=0.8, cur_iter=-1):
#         self.cur_iter = 0 if cur_iter == -1 else cur_iter
#         self.init_beta2 = init_beta2
#         self.c = c
#         self.optimizer = optimizer
#         assert isinstance(
#             optimizer, (torch.optim.Adam, torch.optim.AdamW)
#         ), "should use Adam optimzier, which has beta2"

#     def step(self, cur_iter=None):
#         if cur_iter is None:
#             self.cur_iter += 1
#         else:
#             self.cur_iter = cur_iter

#         new_beta2 = self.get_beta2()
#         import pdb; pdb.set_trace()
#         for pg in self.optimizer.param_groups:
#             beta1, _ = pg["betas"]
#             pg["betas"] = (beta1, new_beta2)

#     def get_beta2(self):
#         if self.c <= 0:
#             return self.init_beta2
#         scale = 1 - (1 / self.cur_iter**self.c)
#         return max(self.init_beta2, scale)