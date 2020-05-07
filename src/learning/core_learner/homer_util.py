import os
import torch
import imageio
import skimage
import collections
import numpy as np

from utils.cuda import cuda_var


class HomerUtil:

    def __init__(self, config, constants):
        self.config = config
        self.constants = constants

    @staticmethod
    def save_homing_policy_figures(env, env_name, homing_policies, step):

        num_samples = 20  # self.constants["eval_homing_policy_sample_size"]
        mixed_image = None
        ctr = 0

        for ix, policy in enumerate(homing_policies[step]):

            if not os.path.exists("./%s_policy/step_%d/" % (env_name, step)):
                os.makedirs("./%s_policy/step_%d/" % (env_name, step))

            mixed_image_step = None
            for j in range(0, num_samples):

                # Rollin for steps
                obs, meta = env.reset()

                for step_ in range(1, step + 1):
                    obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                    action = policy[step_].sample_action(obs_var)
                    obs, reward, done, meta = env.step(action)

                if mixed_image is None:
                    mixed_image = obs
                else:
                    mixed_image = np.maximum(mixed_image, obs)

                if mixed_image_step is None:
                    mixed_image_step = obs
                else:
                    mixed_image_step = np.maximum(mixed_image_step, obs)
                ctr += 1.0

            # Save the observation
            mixed_image_step = skimage.transform.resize(mixed_image_step, (600, 600))
            imageio.imwrite("./%s_policy/step_%d/image_%d.png" %
                              (env_name, step, ix + 1), mixed_image_step)

        mixed_image = skimage.transform.resize(mixed_image, (600, 600))
        imageio.imwrite("./%s_policy/step_%d/mixed_image.png" % (env_name, step), mixed_image)

    @staticmethod
    def save_abstract_state_figures(env_name, observation_samples, step):

        if not os.path.exists("./%s_policy/step_%d/" % (env_name, step)):
            os.makedirs("./%s_policy/step_%d/" % (env_name, step))

        for ix, figures in observation_samples.items():

            max_figure = figures[0]
            for figure in figures:
                max_figure = np.maximum(max_figure, figure)

            imageio.imwrite("./%s_policy/step_%d/abstract_state_%d.png" %
                            (env_name, step, ix), skimage.transform.resize(max_figure, (600, 600)))

    @staticmethod
    def save_newly_explored_states(env_name, dataset, step):

        if not os.path.exists("./%s_policy/step_%d/" % (env_name, step)):
            os.makedirs("./%s_policy/step_%d/" % (env_name, step))

        mixed_image = None
        mixed_image_ix = dict()

        for dp in dataset:
            ix = "None" if dp.get_policy_index() is None else str(dp.get_policy_index())

            if ix in mixed_image_ix:
                mixed_image_ix[ix] = np.maximum(mixed_image_ix[ix], dp.get_next_obs())
            else:
                mixed_image_ix[ix] = dp.get_next_obs()

            if mixed_image is None:
                mixed_image = dp.get_next_obs()
            else:
                mixed_image = np.maximum(mixed_image, dp.get_next_obs())

        # Save the images
        imageio.imwrite("./%s_policy/step_%d/newly_explored_state.png" %
                        (env_name, step), skimage.transform.resize(mixed_image, (600, 600)))

        for ix, image in mixed_image_ix.items():
            imageio.imwrite("./%s_policy/step_%d/explored_state_from_%s.png" %
                            (env_name, step, ix), skimage.transform.resize(image, (600, 600)))

    @staticmethod
    def get_abstract_state_counts(encoding_function, dataset):

        count_stats = {}
        observation_samples = {}  # A collection of 20 observations that map to this value
        for datapoint in dataset:
            if datapoint.is_valid() == 0:
                continue
            obs = datapoint.get_next_obs()
            step = datapoint.get_timestep()

            if "cluster_center" in datapoint.meta_dict:
                ix = datapoint.meta_dict["cluster_center"]
            else:
                ix = encoding_function.encode_observations(obs)

            if ix in count_stats:
                count_stats[ix]["total"] += 1.0
                if step in count_stats[ix]:
                    count_stats[ix][step] += 1.0
                else:
                    count_stats[ix][step] = 1.0
            else:
                count_stats[ix] = {"total": 1.0, step: 1.0}
                observation_samples[ix] = collections.deque(maxlen=20)

            observation_samples[ix].append(obs)

        return count_stats, observation_samples

    def log_homing_policy_reward(self, env, homing_policies, step, logger):

        num_samples = self.constants["eval_homing_policy_sample_size"]
        all_total_reward = 0.0

        for ix, policy in enumerate(homing_policies[step]):

            total_reward = 0.0
            for _ in range(0, num_samples):

                # Rollin for steps
                obs, meta = env.reset()

                for step_ in range(1, step + 1):
                    obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                    action = policy[step_].sample_action(obs_var)
                    obs, reward, done, meta = env.step(action)
                    total_reward = total_reward + reward

            total_reward = total_reward / float(max(1, num_samples))
            all_total_reward = all_total_reward + total_reward
            logger.log("After horizon %r. Policy Number %r receives mean reward %r" % (step, ix + 1, total_reward))

        all_total_reward = all_total_reward / float(max(1, len(homing_policies[step])))
        logger.log("After horizon %r. Random Policy receives reward %r" % (step, all_total_reward))

    def evaluate_homing_policy(self, env, homing_policies, step, logger):

        num_states = {}
        num_samples = self.constants["eval_homing_policy_sample_size"]
        policies_final_observations = []

        for ix, policy in enumerate(homing_policies[step]):

            policy_states = {}
            policy_final_observation = []

            for _ in range(0, num_samples):

                # Rollin for steps
                start_obs, meta = env.reset()
                obs = start_obs

                for step_ in range(1, step + 1):
                    obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
                    action = policy[step_].sample_action(obs_var)
                    obs, reward, done, meta = env.step(action)

                policy_final_observation.append(obs)
                state = str(meta["state"])
                if state in policy_states:
                    policy_states[state] = policy_states[state] + 1
                else:
                    policy_states[state] = 1

                if state in num_states:
                    num_states[state] = num_states[state] + 1
                else:
                    num_states[state] = 1

            policies_final_observations.append(policy_final_observation)

            for state in policy_states:
                policy_states[state] = (policy_states[state] / float(max(1, num_samples))) * 100

            logger.log("After horizon %r. Policy Number %r has distribution %r" % (step, ix + 1, policy_states))

        for state in num_states:
            num_states[state] = (num_states[state] / float(max(1, len(homing_policies[step]) * num_samples))) * 100

        logger.log("After horizon %r. Random Policy distribution is %r" % (step, num_states))

        return num_states, policies_final_observations

    @staticmethod
    def generate_selection_weights(num_homing_policy, num_abstract_states, policies_final_observations,
                                   encoding_function):

        # Create matrix of number of homing policies times the number of encoded states
        correlation = np.zeros((num_homing_policy, num_abstract_states))

        for i in range(0, num_homing_policy):
            policy_final_observations = policies_final_observations[i]

            for obs in policy_final_observations:
                ix = encoding_function.encode_observations(obs)
                correlation[i][ix] += 1.0

        abstract_state_count = correlation.sum(axis=0)
        correlation_score = (correlation/(abstract_state_count + 0.001)).max(axis=1)
        selection_weights = correlation_score/float(max(1.0, correlation_score.sum()))

        return selection_weights

    @staticmethod
    def save_encoder_model(encoding_function, experiment, trial, step, category):

        model_folder_name = experiment + "/trial_%d_encoder_model_%s/" % (trial, category)
        if not os.path.exists(model_folder_name):
            os.makedirs(model_folder_name)
        encoding_function.save(model_folder_name, "encoder_model_%d" % step)
