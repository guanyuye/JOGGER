from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import gym
import torch
from gym.spaces import Discrete, Box, Dict




class CrossVal(gym.Env):
    fold = 0
    process = 0
    def __init__(self, config):
        fold = config['fold_idx'] if "fold_idx" in config.keys() else 0
        proc = 'Eval' if  config['process'] == 1 else 'Train'
        print("Env : ", proc)
        env_name = 'PGsql-{}-Join-Job-v{}'.format(proc, fold)


        self.wrapped = gym.make(env_name)
        self.action_space = self.wrapped.action_space
        self.reward_range = self.wrapped.reward_range

        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(self.wrapped.action_space.n, )),
            "db": self.wrapped.observation_space,
        })

    def update_avail_actions(self):
        self.validActions = self.wrapped.getValidActions()
        self.action_mask = np.array([0.] * self.action_space.n)
        for i in self.validActions:
            self.action_mask[i] = 1

    def reset(self):
        obs,table_num = self.wrapped.reset()
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "db": obs,
        }, table_num

    def step(self, action):
        if action not in self.validActions:
            import pdb;pdb.set_trace()
        #     print("failure steps:")

        orig_obs, rew, done, sub_trees,join_num= self.wrapped.step(action)

        self.update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "db": orig_obs,
        }
        return obs, rew, done, sub_trees,join_num



