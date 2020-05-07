import os
import time
import pickle
import numpy as np

from environments.environment_package.noise_gen import generated_hadamhard_matrix


class Environment(object):
    """
    An environment skeleton. Defaults to simple MAB
    H = 1, K=2, rewards are bernoulli, sampled from dirichlet([1,1]) prior.
    """

    BERNOULLI, GAUSSIAN, HADAMHARD, HADAMHARDG = range(4)

    def __init__(self):
        self.state = None
        self.h = 0

    def start_episode(self):
        self.h = 0
        self.state = self.start()
        return self.make_obs(self.state), {"state": None if self.state is None else tuple(self.state)}

    def get_actions(self):
        if self.state is None:
            raise Exception("Episode not started")
        if self.h == self.horizon:
            return None
        return self.actions
    
    def make_obs(self, s):
        return s

    def act(self, a):
        if self.state is None:
            raise Exception("Episode not started")

        if self.h == self.horizon:
            raise Exception("Cannot execute more than actions than horizon (%d)" % self.horizon)
        else:
            new_state = self.transition(self.state, a)
            self.h += 1

        r = self.reward(self.state, a, new_state)
        self.state = new_state

        # Create a dictionary containing useful debugging information
        info = {"state": None if self.state is None else tuple(self.state)}

        return self.make_obs(self.state), r, info

    def get_num_actions(self):
        return len(self.actions)

    def is_tabular(self):
        return True

    def get_dimension(self):
        assert not self.is_tabular(), "Not a featurized environment"
        return self.dim


class DiabolicalCombinationLock(Environment):

    def __init__(self, horizon, swap, num_actions, anti_shaping_reward, noise_type,
                 optimal_reward=1.0, anti_shaping_reward2=1.0, seed=1):
        """
        :param horizon: Horizon of the MDP
        :param swap: Probability for stochastic edges
        :param noisy_dim: Dimension of noise
        :param noise_type: Type of Noise
        """

        Environment.__init__(self)
        self.horizon = horizon
        self.swap = swap
        self.noise_type = noise_type
        self.num_actions = num_actions
        self.optimal_reward = optimal_reward
        self.optimal_reward_prob = 1.0
        self.rng = np.random.RandomState(seed)

        assert anti_shaping_reward < self.optimal_reward * self.optimal_reward_prob, \
            "Anti shaping reward shouldn't exceed optimal reward which is %r" % \
            (self.optimal_reward * self.optimal_reward_prob)

        self.anti_shaping_reward = anti_shaping_reward
        self.anti_shaping_reward2 = anti_shaping_reward2

        assert num_actions >= 2, "Atleast two actions are needed"
        self.actions = list(range(0, num_actions))

        self.opt_a = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)
        self.opt_b = self.rng.randint(low=0, high=self.num_actions, size=self.horizon)

        if noise_type == Environment.GAUSSIAN:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            self.dim = self.horizon + 4

        elif noise_type == Environment.BERNOULLI:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1. We further add noise of size horizon.
            self.dim = self.horizon + 4 + self.horizon  # Add noise of length horizon

        elif noise_type == Environment.HADAMHARD:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        elif noise_type == Environment.HADAMHARDG:

            # We encode the state type and time separately. The type is one of the 3 and the time could be any value
            # in 1 to horizon + 1.
            lower_bound = self.horizon + 4
            self.hadamhard_matrix = generated_hadamhard_matrix(lower_bound)
            self.dim = self.hadamhard_matrix.shape[0]

        else:
            raise AssertionError("Unhandled noise type %r" % self.noise_type)

    def transition(self, x, a):

        if x is None:
            raise Exception("Not in any state")

        b = self.rng.binomial(1, self.swap)

        if x[0] == 0 and a == self.opt_a[x[1]]:
            if b == 0:
                return [0, x[1] + 1]
            else:
                return [1, x[1] + 1]
        if x[0] == 1 and a == self.opt_b[x[1]]:
            if b == 0:
                return [1, x[1] + 1]
            else:
                return [0, x[1] + 1]
        else:
            return [2, x[1] + 1]

    def make_obs(self, x):

        if x is None or self.dim is None:
            return x
        else:

            if self.noise_type == Environment.BERNOULLI:

                v = np.zeros(self.dim, dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v[self.horizon + 4:] = self.rng.binomial(1, 0.5, self.horizon)

            elif self.noise_type == Environment.GAUSSIAN:

                v = np.zeros(self.dim, dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)

            elif self.noise_type == Environment.HADAMHARD:

                v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v = np.matmul(self.hadamhard_matrix, v)

            elif self.noise_type == Environment.HADAMHARDG:

                v = np.zeros(self.hadamhard_matrix.shape[1], dtype=float)
                v[x[0]] = 1.0
                v[3 + x[1]] = 1.0
                v = v + self.rng.normal(loc=0.0, scale=0.1, size=v.shape)
                v = np.matmul(self.hadamhard_matrix, v)

            else:
                raise AssertionError("Unhandled noise type %r" % self.noise_type)

            return v

    def start(self):
        # Start stochastically in one of the two live states
        toss_value = self.rng.binomial(1, 0.5)

        if toss_value == 0:
            return [0, 0]
        elif toss_value == 1:
            return [1, 0]
        else:
            raise AssertionError("Toss value can only be 1 or 0. Found %r" % toss_value)

    def reward(self, x, a, next_x):

        # If the agent reaches the final live states then give it the optimal reward.
        if (x == [0, self.horizon - 1] and a == self.opt_a[x[1]]) or \
                (x == [1, self.horizon - 1] and a == self.opt_b[x[1]]):
            return self.optimal_reward * self.rng.binomial(1, self.optimal_reward_prob)

        # If reaching the dead state for the first time then give it a small anti-shaping reward.
        # This anti-shaping reward is anti-correlated with the optimal reward.
        if x is not None and next_x is not None:
            if x[0] != 2 and next_x[0] == 2:
                return self.anti_shaping_reward * self.rng.binomial(1, 0.5)
            elif x[0] != 2 and next_x[0] != 2:
                return - self.anti_shaping_reward2 / (self.horizon - 1)

        return 0

    def get_optimal_value(self):
        # TODO: HARDCODING FOR MIKAEL's ENVIRONMENT. REMOVE IN FUTURE
        return 4.0
        # return self.optimal_reward * self.optimal_reward_prob

    def is_tabular(self):
        return self.dim is None

    def save(self, folder_name):
        """ Save the environment given the folder name """

        timestamp = time.time()

        if not os.path.exists(folder_name + "/env_%d" % timestamp):
            os.makedirs(folder_name + "/env_%d" % timestamp, exist_ok=True)

        with open(folder_name + "/env_%d/diabcombolock" % timestamp, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(env_folder_name):
        """ Load the environment from the environment folder name """

        with open(env_folder_name + "/diabcombolock", "rb") as f:
            env = pickle.load(f)

        return env


def run_rl_acid_environment(env_name):

    if env_name == 'MAB':
        E = Environment()
        rewards = [0,0]
        counts = [0,0]
        for t in range(1000):
            x = E.start_episode()
            while x is not None:
                actions = E.get_actions()
                a = np.random.choice(actions)
                (x,r) = E.act(a)
                rewards[a] += r
                counts[a] += 1
        for a in [0,1]:
            assert (np.abs(np.float(rewards[a])/counts[a] -E.reward_dists[a]) < 0.1)

    else:
        raise AssertionError("Environment name not found %r" % env_name)
