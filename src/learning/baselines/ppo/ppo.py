#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
#  Dear Reviewers. The code is anonymized and based on the original code
#  provided by Shangtong Zhang. Authors are in *no manner* related to Shangtong Zhang.
#  The above declaration is required for building upon their codebase.
#######################################################################

import torch
import torch.nn as nn
import numpy as np

from learning.baselines.ppo.base_agent import BaseAgent
from utils.cuda import cuda_var
from learning.baselines.ppo.ppo_model import PPOModel


class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def to_np(t):
    return t.cpu().numpy()


class PPOAgent(BaseAgent):

    def __init__(self, ppo_config, config, constants):
        BaseAgent.__init__(self, config)
        self.ppo_config = ppo_config
        self.model = PPOModel(config, constants)
        self.opt = torch.optim.RMSprop(self.model.parameters(), 0.001)

    def step(self, env, num_episodes_per_update, logger):
        """ Collect some samples using the current model and update """

        collected_reward = 0.0
        ppo_config = self.ppo_config

        states_ls, actions_ls, log_probs_old_ls, returns_ls, advantages_ls = [], [], [], [], []

        for _ in range(0, num_episodes_per_update):

            storage = Storage(ppo_config.rollout_length)
            states, _ = env.reset()
            for _ in range(ppo_config.rollout_length):
                states_var = cuda_var(torch.from_numpy(states).float()).view(1, -1)
                prediction = self.model.forward(states_var)
                next_states, rewards, terminals, info = env.step(to_np(prediction['a'].data))
                collected_reward += rewards
                terminals = 1 if terminals else 0
                storage.add(prediction)
                storage.add({'r': cuda_var(torch.from_numpy(np.array([rewards])).float()).unsqueeze(0),
                             'm': cuda_var(torch.from_numpy(np.array([1 - terminals])).float()).unsqueeze(0),
                             's': cuda_var(torch.from_numpy(states)).view(1, -1)})
                states = next_states

            states_var = cuda_var(torch.from_numpy(states).float()).view(1, -1)
            prediction = self.model.forward(states_var)
            storage.add(prediction)
            storage.placeholder()

            advantages = cuda_var(torch.from_numpy(np.zeros((1, 1))).float())
            returns = prediction['v']
            for i in reversed(range(ppo_config.rollout_length)):
                returns = storage.r[i] + ppo_config.discount * storage.m[i] * returns
                if not ppo_config.use_gae:
                    advantages = returns - storage.v[i]
                else:
                    td_error = storage.r[i] + ppo_config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                    advantages = advantages * ppo_config.gae_tau * ppo_config.discount * storage.m[i] + td_error
                storage.adv[i] = advantages.detach()
                storage.ret[i] = returns.detach()

            states_, actions_, log_probs_old_, returns_, advantages_ = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
            states_ls.append(states_)
            actions_ls.append(actions_)
            log_probs_old_ls.append(log_probs_old_)
            returns_ls.append(returns_)
            advantages_ls.append(advantages_)

        states = torch.cat(states_ls, dim=0)
        actions =  torch.cat(actions_ls, dim=0)
        log_probs_old = torch.cat(log_probs_old_ls, dim=0)
        returns = torch.cat(returns_ls, dim=0)
        advantages = torch.cat(advantages_ls, dim=0)

        # print("States ", states.size())
        # print("Actions ", actions.size())
        # print("Log Probs Old ", log_probs_old.size())
        # print("Returns ", returns.size())
        # print("Advantages ", advantages.size())

        states = states.float()
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        if advantages.size(0) == 1:
            advantages = advantages * 0.0
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 0.000001)

        for _ in range(ppo_config.optimization_epochs):

            sampler = random_sample(np.arange(states.size(0)), ppo_config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = cuda_var(torch.from_numpy(batch_indices).long())
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.model.forward(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.ppo_config.ppo_ratio_clip,
                                          1.0 + self.ppo_config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - ppo_config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm(self.model.parameters(), ppo_config.gradient_clip)
                self.opt.step()

        return collected_reward
