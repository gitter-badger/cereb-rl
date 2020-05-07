# This is a self contained code for testing on simple continuous state space problem
# This file is not supposed to be a fully supported library for applying Homer to continuous state problem

import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from utils.cuda import cuda_var
from utils.gumbel import gumbel_sample


class MDP(gym.Env):
    """ A simple 1 D and horizon 1 Continuous State Space problem """

    def __init__(self, state_dim, action_dim, horizon, action_range, state_space_range):
        """ """

        if state_dim != 1 or action_dim != 1 or horizon != 1:
            raise NotImplementedError()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_range = action_range
        self.state_space_range = state_space_range

        self.offset = 0.0  # np.random.uniform(low=self.state_space_range[0], high=self.state_space_range[1])

    def step(self, action):
        """ """

        state_space_dim = self.state_space_range[1] - self.state_space_range[0]
        mean = (self.state_space_range[0] + self.state_space_range[1]) / 2.0

        # Sample a Gaussian Noise
        noise = np.random.normal(loc=mean, scale=0.5)

        new_state = (self.curr_state + action + self.offset + noise - self.state_space_range[0]) % (state_space_dim) + self.state_space_range[0]

        print("%r ->  %r, Noise %r, Offset %r" % (self.curr_state, new_state, noise, self.offset))

        reward = 0.0
        done = True
        info = {}

        self.curr_state = new_state

        return self.curr_state, reward, done, info

    def reset(self):
        """ """
        self.curr_state = 0.0
        return self.curr_state  # np.random.uniform(low=self.state_space_range[0], high=self.state_space_range[1])

    def render(self, mode='human'):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class Model(nn.Module):

    def __init__(self, state_dim, act_dim, act_emb_dim, N):
        super(Model, self).__init__()

        self.obs_layer = nn.Linear(state_dim, N)
        self.action_layer = nn.Linear(act_dim, act_emb_dim)

        self.phi_embedding = nn.Embedding(N, N)
        self.phi_embedding.weight.data.copy_(torch.from_numpy(np.eye(N)).float())
        self.phi_embedding.weight.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(state_dim + act_dim + N, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 2)
        )

    def forward(self, curr_obs, actions, next_obs):

        action_x = self.action_layer(actions)

        x = self.obs_layer(next_obs)
        prob, log_prob = gumbel_sample(x, temperature=1)
        expected_vector = torch.matmul(prob, self.phi_embedding.weight)
        x = torch.cat([curr_obs, action_x, expected_vector], dim=1)
        logits = self.classifier(x)

        return F.log_softmax(logits, dim=1)

    def calc_loss(self, batch):

        curr_obs = cuda_var(torch.cat([torch.from_numpy(np.array(point[0])).view(1, -1)
                                                for point in batch], dim=0)).float()
        actions = cuda_var(torch.cat([torch.from_numpy(np.array(point[1])).view(1, -1)
                                      for point in batch], dim=0)).float()
        next_obs = cuda_var(torch.cat([torch.from_numpy(np.array(point[2])).view(1, -1)
                                           for point in batch], dim=0)).float()
        gold_labels = cuda_var(torch.cat([torch.from_numpy(np.array(point[3])).view(1, -1)
                                          for point in batch], dim=0)).long()

        log_probs = self.forward(curr_obs, actions, next_obs)
        classification_loss = -torch.mean(log_probs.gather(1, gold_labels.view(-1, 1)))

        return classification_loss

    def encode_obs(self, batch):

        next_obs = cuda_var(torch.cat([torch.from_numpy(np.array(point[2])).view(1, -1)
                                       for point in batch], dim=0)).float()

        log_prob = F.log_softmax(self.obs_layer(next_obs), dim=1).view(len(batch), -1)
        return log_prob.max(1)[1]

# Run Homer
config = {
   "num_samples": 1000,
    "state_dim": 1,
    "action_dim": 1,
    "horizon": 1,
    "state_range": [-1, 1],
    "action_range": [-1, 1],
    "N": 20,
    "batch_size": 32,
    "learning_rate": 0.01,
    "max_epoch": 1000
}

env = MDP(state_dim=config["state_dim"],
          action_dim=config["action_dim"],
          horizon=config["horizon"],
          action_range=config["state_range"],
          state_space_range=config["action_range"])

pos_data = []
for n in range(config["num_samples"]):
    obs = env.reset()
    # Take a ranodmly chosen uniform action
    action = np.random.uniform(low=-1, high=1)
    new_obs, reward, done, info = env.step(action)
    pos_data.append((obs, action, new_obs, 1))
num_pos = len(pos_data)

# Create negative examples
neg_data = []
for dp in pos_data:
    neg_dp = (dp[0], dp[1], pos_data[random.randint(0, num_pos - 1)][2], 0)
    neg_data.append(neg_dp)

data = pos_data
data.extend(neg_data)
random.shuffle(data)

######################
# x_pc = [dp[1] for dp in data if dp[3] == 1]
# y_pc = [0] * len(x_pc)
#
# x_pn = [dp[2] for dp in data if dp[3] == 1]
# y_pn = [1] * len(x_pn)
#
# x_nc = [dp[1] for dp in data if dp[3] == 0]
# y_nc = [0] * len(x_nc)
#
# x_nn = [dp[2] for dp in data if dp[3] == 0]
# y_nn = [1] * len(x_nn)
#
#
# # Create plot
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# ax.scatter(x_pc, y_pc, alpha=0.8, c="red", edgecolors='none', s=30)
# ax.scatter(x_pn, y_pn, alpha=0.8, c="red", edgecolors='none', s=30)
#
# ax.scatter(x_nc, y_nc, alpha=0.8, c="blue", edgecolors='none', s=30)
# ax.scatter(x_nn, y_nn, alpha=0.8, c="blue", edgecolors='none', s=30)
#
# plt.title('Matplot scatter plot')
# plt.legend(loc=2)
# plt.show()
######################


# Create the model and optimizer
model = Model(state_dim=config["state_dim"],
              act_dim=config["action_dim"],
              act_emb_dim=config["action_dim"],
              N=config["N"])

optimizer = optim.Adam(params=model.parameters(), lr=config["learning_rate"])

# Train the model on Homer
num_train = int(0.8 * len(data))
train_data = data[:num_train]
test_data = data[num_train:]
train_data_size = len(train_data)

best_test_loss = float('inf')
for epoch in range(1, config["max_epoch"] + 1):

    batches = [train_data[i:i + config["batch_size"]] for i in range(0, train_data_size, config["batch_size"])]

    for batch in batches:

        loss = model.calc_loss(batch)

        # Do update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        optimizer.step()

    # Compute train and test loss
    train_loss = 0.0
    for batch in batches:
        train_loss += float(model.calc_loss(batch))

    test_loss = float(model.calc_loss(test_data))
    best_test_loss = min(best_test_loss, test_loss)

    print("Epoch %d, Train Loss: %r, Test Loss: %r, Best Test Loss %r" % (epoch, train_loss, test_loss, best_test_loss))

# Visualize the result
mapping = dict()
for i in range(config["N"]):
    mapping[i] = []

dataset_size = len(data)
batches = [data[i:i + config["batch_size"]] for i in range(0, dataset_size, config["batch_size"])]
for batch in batches:

    encoding = model.encode_obs(batch)
    print(encoding.size())
    for j, dp in enumerate(batch):
        mapping[int(encoding[j])].append(dp)

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = ["r", "b", "g", "k", "c", "m", "y"]
color_ix = 0

for i in range(0, config["N"]):
    data = mapping[i]

    if len(data) > 0:

        x = [dp[2] for dp in data]
        y = [0] * len(x)

        ax.scatter(x, y, alpha=0.8, c=colors[color_ix], edgecolors='none', s=30, label="Abstract State %d" % i)
        color_ix = (color_ix + 1) % len(colors)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()


