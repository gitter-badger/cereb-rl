import os
import time
import pickle
import numpy as np
import torch.multiprocessing as mp
import learning.learning_utils.policy_evaluate as policy_evaluate

from learning.core_learner.policy_search_wrapper import PolicySearchWrapper
from learning.learning_utils.encoder_sampler_wrapper import EncoderSamplerWrapper
from learning.learning_utils.train_encoding_function import TrainEncodingFunction
from learning.core_learner.homer_util import HomerUtil
from model.encoder_model_wrapper import EncoderModelWrapper
from utils.tensorboard import Tensorboard


class DistributedHomerAlgorithm:
    """ A novel model-free off-policy reinforcement learning algorithm with PAC guarantees and polynomial
      time complexity for reinforcement learning in rich observation settings. """

    def __init__(self, config, constants):

        self.config = config
        self.constants = constants

        # Train encoding function
        self.train_encoding_function = TrainEncodingFunction(config, constants)

        # Sampler for generating data for training the encoding function
        self.encoder_sampler = EncoderSamplerWrapper(constants)

        # Policy Search Routines. HOMER can use different algorithms to explore in the reward-free
        # setting and in the reward-sensitive setting.
        self.reward_free_planner = PolicySearchWrapper.generate_policy_search(
            constants["reward_free_planner"], config, constants)

        self.reward_sensitive_planner = PolicySearchWrapper.generate_policy_search(
            constants["reward_sensitive_planner"], config, constants)

        # Util object
        self.util = HomerUtil(config, constants)

    def find_abstract_states_to_explore(self, count_stats, num_state_budget, step):

        abstract_states_to_explore = []
        for ix in range(0, num_state_budget):

            add_state = True

            if self.constants["filter_unreachable_abstract_states"]:
                # Check if the abstract state is reachable at the new time step. This is done by checking
                # the number of observations at the new time step that were mapped to this abstract state.
                if ix not in count_stats or count_stats[ix]["total"] == 0:
                    add_state = False

            if self.constants["filter_old_abstract_states"]:
                # Check if the abstract state was reached at previous time steps. If this is the case then
                # this abstract state has already been explored and, therefore, it should not be explored.
                # This flag only makes sense when we are doing dataset aggregation across time step.
                if ix not in count_stats or step not in count_stats[ix] \
                        or count_stats[ix]["total"] - count_stats[ix][step] > 0.0:
                    add_state = False

            if add_state:
                abstract_states_to_explore.append(ix)

        return abstract_states_to_explore

    def single_process_ps(self, env, actions, step, replay_memory, homing_policies, useful_abstract_states,
                          tensorboard, encoding_function, logger, use_pushover):
        """ Learn a homing policy using the main process """

        # Fetch part of replay memory that can be used by the reward_free_planner.
        filtered_dataset = PolicySearchWrapper.get_filtered_data(replay_memory, step, self.reward_free_planner)

        # Learn homing policies using the trained classifier
        for i in useful_abstract_states:

            logger.log("Learning homing policy to reach abstraction number %d " % i)

            # Reward function incentivizes reaching the frontier and achieving the right output for the encoding fn.
            reward_func = lambda obs, time: 1 if time == step and \
                                                 encoding_function.encode_observations(obs) == i else 0
            homing_policy, mean_reward, _ = self.reward_free_planner.train(filtered_dataset, env, actions, step,
                                                                           reward_func, homing_policies, logger,
                                                                           tensorboard, False, use_pushover, i)

            homing_policies[step].append(homing_policy)

    def multi_processing_ps(self, experiment, env, env_name, actions, step, replay_memory, homing_policies,
                            useful_abstract_states, num_processes, encoding_function, logger, use_pushover, trial):
        """ Learn a homing policy using the multiple processes """

        # Fetch part of replay memory that can be used by the reward_free_planner.
        filtered_dataset = PolicySearchWrapper.get_filtered_data(replay_memory, step, self.reward_free_planner)

        # Learn homing policies using the trained classifier in parallel
        reward_batches = [useful_abstract_states[i:i + num_processes]
                          for i in range(0, len(useful_abstract_states), num_processes)]

        assert env.is_thread_safe(), "To bootstrap it must be thread safe"
        env_info = (env_name, env.get_bootstrap_env())

        for reward_batch in reward_batches:

            processes = []
            for reward_id in reward_batch:
                logger.log("Learning homing policy to reach abstraction number %d " % reward_id)

                # Reward args contain the arguments for homing policy reward function
                reward_args = (encoding_function, reward_id)

                policy_folder_name = experiment + "/trial_%d_horizon_%d_homing_policy_%d/" % (trial, step, reward_id)
                p = mp.Process(target=self.reward_free_planner.do_train,
                               args=(self.config, self.constants, filtered_dataset, env_info, policy_folder_name,
                                     actions, step, reward_args, homing_policies, logger, False, use_pushover))

                p.daemon = False
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        # Read all the learned policies from disk
        for i in useful_abstract_states:
            policy_folder_name = experiment + "/trial_%d_horizon_%d_homing_policy_%d/" % (trial, step, i)
            previous_step_homing_policy = None if step == 1 else homing_policies[step - 1]
            policy = self.reward_free_planner.read_policy(policy_folder_name, step,
                                                          previous_step_homing_policy, delete=False)
            # TODO add step for filtering failed policies
            homing_policies[step].append(policy)

    def train(self, experiment, env, env_name, num_processes, experiment_name, logger, use_pushover,
              debug, homing_policy_validation_fn, trial=1, do_reward_sensitive_learning=False):
        """ Execute HOMER algorithm on an environment using
        :param experiment:
        :param env:
        :param env_name:
        :param num_processes:
        :param experiment_name:
        :param logger:
        :param use_pushover: True/False based on whether pushover is used
        :param debug:
        :param homing_policy_validation_fn:
        :param trial:
        :param do_reward_sensitive_learning:
        :return:
        """

        horizon = self.config["horizon"]
        actions = self.config["actions"]
        num_samples = self.constants["encoder_training_num_samples"]
        tensorboard = Tensorboard(log_dir=self.config["save_path"])

        homing_policies = dict()    # Contains a set of homing policies for every time step
        encoding_function = None    # Learned encoding function for the current time step
        dataset = []                # Dataset of samples collected for training the encoder
        replay_memory = dict()      # Replay memory of *all* deviation transitions indexed by time step

        for step in range(1, horizon + 1):

            logger.log("Running Homer: Step %r out of %r " % (step, horizon))

            homing_policies[step] = []      # Homing policies for this time step
            replay_memory[step] = []        # Replay memory for this time step

            # Step 1: Create dataset for learning the encoding function. A single datapoint consists of a transition
            # (x, a, x') and a 0-1 label y. If y=1 then transition was observed and y=0 otherwise.
            time_collection_start = time.time()
            dataset = self.encoder_sampler.gather_samples(env, actions, step, homing_policies, num_samples, dataset)
            replay_memory[step] = [dp for dp in dataset if dp.is_valid() == 1 and dp.get_timestep() == step]
            logger.log("Encoder: %r samples collected in %r sec" % (num_samples, time.time() - time_collection_start))

            # Step 2: Perform binary classification on the dataset. The classifier f(x, a, x') is trained to predict
            # the probability that a transition (x, a, x') was observed. There are two type of classifiers that we
            # support. The first classifier has an internal bottleneck feature that allows for recovering state
            # abstraction function while other performs clustering on top of a train model without discretization.
            time_encoder_start = time.time()

            encoding_function, num_state_budget = self.train_encoding_function.do_train(
                dataset, logger, tensorboard, debug, bootstrap_model=encoding_function,
                undiscretized_initialization=True, category="backward")

            self.util.save_encoder_model(encoding_function, experiment, trial, step, "backward")
            logger.log("Encoder: Training time %r" % (time.time() - time_encoder_start))

            # Step 3: Find which abstract states should be explored. This is basically done based on which
            # abstract states have a non-zero count. Example, one can specify a really high budget for abstract
            # states but most of them are never used. This is not a problem when using the clustering oracle.
            count_stats, observation_samples = self.util.get_abstract_state_counts(encoding_function, dataset)
            abstract_states_to_explore = self.find_abstract_states_to_explore(count_stats, num_state_budget, step)
            logger.log("Abstract State by Counts: %r" % count_stats)
            logger.debug("Abstract States to explore %r" % abstract_states_to_explore)

            # Step 4: Learn homing policies by planning to reach different abstract states
            if num_processes == 1:  # Single process needed. Run it on the current process.
                self.single_process_ps(env, actions, step, replay_memory, homing_policies, abstract_states_to_explore,
                                       tensorboard, encoding_function, logger, use_pushover)
            else:
                self.multi_processing_ps(experiment, env, env_name, actions, step, replay_memory, homing_policies,
                                         abstract_states_to_explore, num_processes, encoding_function,
                                         logger, use_pushover, trial)
            logger.log("Homer step %r took time %r" % (step, time.time() - time_collection_start))

            # Step 5 (Optional): Automatic evaluation of homing policies if possible. A validation function can
            # check if homing policy has good coverage over the underline state.
            if homing_policy_validation_fn is not None:

                state_dist, _ = self.util.evaluate_homing_policy(env, homing_policies, step, logger)

                if not homing_policy_validation_fn(state_dist, step):
                    logger.log("Didn't find a useful policy cover for step %r" % step)
                    return policy_evaluate.generate_failure_result(env, env.num_eps)
                else:
                    logger.log("Found useful policy cover for step %r " % step)

            # Step 6 (Optional): Performing debugging based on learned state abstraction and
            # policy cover for this time step.
            if debug:
                # Log the environment reward received by the policy
                self.util.log_homing_policy_reward(env, homing_policies, step, logger)

                if self.config["feature_type"] == "image":
                    # For environments generating image, it is often not possible to get access to the underline state
                    # therefore we save images for debugging.
                    self.util.save_homing_policy_figures(env, env_name, homing_policies, step)

                    # Save the abstract state and an image
                    if observation_samples is not None:
                        self.util.save_abstract_state_figures(env_name, observation_samples, step)

                    # Save newly explored states
                    self.util.save_newly_explored_states(env_name, dataset, step)

        if not do_reward_sensitive_learning:

            return dict()
        else:

            logger.log("Reward Sensitive Learning: Computing the optimal policy for the environment reward function")

            # Compute the optimal policy
            reward_planning_start_time = time.time()
            approx_optimal_policy, _, info = self.reward_sensitive_planner.train(replay_memory=replay_memory,
                                                                                 env=env,
                                                                                 actions=actions,
                                                                                 horizon=horizon,
                                                                                 reward_func=None,
                                                                                 homing_policies=homing_policies,
                                                                                 logger=logger,
                                                                                 tensorboard=tensorboard,
                                                                                 debug=True,
                                                                                 use_pushover=use_pushover)
            logger.log("Reward Sensitive Learning: Time %r" % (time.time() - reward_planning_start_time))

            logger.log("Actual: Total number of episodes used %d. Total return %f." %
                       (env.num_eps, env.sum_total_reward))

            # Evaluate the optimal policy
            return policy_evaluate.evaluate(env, approx_optimal_policy, horizon, logger,
                                            env.num_eps, env.sum_total_reward)

    def train_from_learned_homing_policies(self, env, load_folder, train_episodes,
                                           experiment_name, logger, use_pushover, trial=1):

        horizon = self.config["horizon"]
        actions = self.config["actions"]
        num_state_budget = self.constants["num_homing_policy"]
        logger.log("Training episodes %d" % train_episodes)

        tensorboard = Tensorboard(log_dir=self.config["save_path"])

        homing_policies = dict()  # Contains a set of homing policies for every time step

        # Load homing policy from folder
        logger.log("Loading Homing policies...")
        for step in range(1, horizon + 1):

            homing_policies[step] = []

            for i in range(0, num_state_budget):
                # TODO can fail if the policy doesn't exist. Add checks to prevent that.
                policy_folder_name = load_folder + "/trial_%d_horizon_%d_homing_policy_%d/" % (trial, step, i)
                if not os.path.exists(policy_folder_name):
                    logger.log("Did not find %s" % policy_folder_name)
                    continue
                previous_step_homing_policy = None if step == 1 else homing_policies[step - 1]
                policy = self.reward_free_planner.read_policy(policy_folder_name, step, previous_step_homing_policy)
                homing_policies[step].append(policy)
        logger.log("Loaded Homing policy.")

        logger.log("Reward Sensitive Learning: Computing the optimal policy for the given reward")

        # Compute the optimal policy
        psdp_start = time.time()
        approx_optimal_policy, _, info = self.reward_sensitive_planner.train(None, env, actions, horizon, None, homing_policies,
                                                                             logger, tensorboard, True, use_pushover)
        logger.log("PSDP Time %r" % (time.time() - psdp_start))

        train_episodes = train_episodes + info["total_episodes"]
        train_reward = info["sum_rewards"]

        # Evaluate the optimal policy
        return policy_evaluate.evaluate(env, approx_optimal_policy, horizon, logger, train_episodes, train_reward)

    def learn_model(self, env, load_folder, experiment_name, experiment, logger, use_pushover, trial=1):

        horizon = self.config["horizon"]
        actions = self.config["actions"]
        num_samples = self.constants["encoder_training_num_samples"]
        num_state_budget = self.constants["num_homing_policy"]

        tensorboard = Tensorboard(log_dir=self.config["save_path"])

        homing_policies = dict()  # Contains a set of homing policies for every time step

        # Load homing policy from folder
        logger.log("Loading Homing policies...")
        for step in range(1, horizon + 1):

            homing_policies[step] = []

            for i in range(0, num_state_budget):
                # TODO can fail if the policy doesn't exist. Add checks to prevent that.
                policy_folder_name = load_folder + "/trial_%d_horizon_%d_homing_policy_%d/" % (trial, step, i)
                if not os.path.exists(policy_folder_name):
                    logger.log("Did not find %s" % policy_folder_name)
                    continue
                previous_step_homing_policy = None if step == 1 else homing_policies[step - 1]
                policy = self.reward_free_planner.read_policy(policy_folder_name, step, previous_step_homing_policy)
                homing_policies[step].append(policy)
        logger.log("Loaded Homing policy.")

        # Load the encoder models
        backward_models = dict()
        backward_models[1] = None
        for step in range(1, horizon + 1):
            backward_model = EncoderModelWrapper.get_encoder_model(self.constants["model_type"], self.config, self.constants)
            backward_model.load(load_folder + "/trial_%d_encoder_model/" % trial, "encoder_model_%d" % step)
            backward_models[step + 1] = backward_model

        encoding_function = None    # Learned encoding function for the current time step
        dataset = []                # Dataset of samples collected for training the encoder
        selection_weights = None    # A distribution over homing policies from the previous time step (can be None)

        # Learn Forward Model and Estimate the Model
        forward_models = dict()
        forward_models[horizon + 1] = None
        prev_dataset = None

        for step in range(1, horizon + 1):

            logger.log("Step %r out of %r " % (step, horizon))

            # Step 1: Create dataset for learning the encoding function. A single datapoint consists of a transition
            # (x, a, x') and a 0-1 label y. If y=1 then transition was observed and y=0 otherwise.
            time_collection_start = time.time()
            dataset = self.encoder_sampler.gather_samples(env, actions, step, homing_policies, num_samples,
                                                          dataset, selection_weights)
            logger.log("Encoder: %r sample collected in %r sec" % (num_samples, time.time() - time_collection_start))

            # Step 2: Train a binary classifier on this dataset. The classifier f(x, a, x') is trained to predict
            # the probability that the transition (x, a, x') was observed. Importantly, the classifier has a special
            # structure f(x, a, x') = p(x, a, \phi(x')) where \phi maps x' to a set of discrete values.
            time_encoder_start = time.time()
            if not self.constants["bootstrap_encoder_model"]:
                encoding_function = None
            encoding_function, _ = self.train_encoding_function.do_train_with_discretized_models(
                dataset, logger, tensorboard, False, bootstrap_model=encoding_function,
                undiscretized_initialization=True, category="forward")
            self.util.save_encoder_model(encoding_function, experiment, trial, step, "forward")
            forward_models[step] = encoding_function
            logger.log("Encoder: Training time %r" % (time.time() - time_encoder_start))

            if step > 1:

                self._estimate_and_save_transition_dynamics(env, experiment, prev_dataset, step,
                                                            forward_models[step - 1],
                                                            backward_models[step - 1],
                                                            forward_models[step],
                                                            backward_models[step],
                                                            logger, trial)

            prev_dataset = dataset

    @staticmethod
    def _estimate_and_save_transition_dynamics(env, experiment, dataset, step, forward_curr, backward_curr,
                                               forward_next, backward_next, logger, trial=1):

        # Store correlations between states and assigned states
        forward_curr_state_corr = dict()
        backward_curr_state_corr = dict()
        ki_curr_state_corr = dict()

        forward_next_state_corr = dict()
        backward_next_state_corr = dict()
        ki_next_state_corr = dict()

        # Estimate the model for the last time step
        transition_model = dict()  # Dictionary of dictionary
        keys = set()
        next_keys = set()

        curr_states = [(0, step - 2), (1, step - 2), (2, step - 2)]
        next_states = [(0, step - 1), (1, step - 1), (2, step - 1)]

        # TODO: Currently debugging makes the code specific to DiabComboLock
        for point in dataset:

            if not point.is_valid():  # Do not learn data on wrong transitions
                continue

            forward_curr_ix = 1 if forward_curr is None else forward_curr.encode_observations(point.get_curr_obs())
            backward_curr_ix = 1 if backward_curr is None else backward_curr.encode_observations(point.get_curr_obs())

            forward_next_ix = 1 if forward_next is None else forward_next.encode_observations(point.get_next_obs())
            backward_next_ix = 1 if backward_next is None else backward_next.encode_observations(point.get_next_obs())

            def update(my_dict, my_key, my_val):
                if my_key in my_dict:
                    my_dict[my_key]["total"] += 1
                    my_dict[my_key]["data"][my_val] += 1
                else:
                    data_ = np.array([0, 0, 0])
                    data_[my_val] = 1.0
                    my_dict[my_key] = {"total": 1, "data": data_}

            curr_index = curr_states.index(point.get_curr_state())
            next_index = next_states.index(point.get_next_state())

            update(forward_curr_state_corr, forward_curr_ix, curr_index)
            update(backward_curr_state_corr, backward_curr_ix, curr_index)
            update(ki_curr_state_corr, (forward_curr_ix, backward_curr_ix), curr_index)

            update(forward_next_state_corr, forward_next_ix, next_index)
            update(backward_next_state_corr, backward_next_ix, next_index)
            update(ki_next_state_corr, (forward_next_ix, backward_next_ix), next_index)

            action = point.get_action()

            key = "state (%d, %d), action %d" % (forward_curr_ix, backward_curr_ix, action)
            next_key = "(%d, %d)" % (forward_next_ix, backward_next_ix)

            keys.add(key)
            next_keys.add(next_key)

            if key in transition_model:
                if next_key in transition_model[key]:
                    transition_model[key][next_key] += 1.0
                else:
                    transition_model[key][next_key] = 1.0
            else:
                transition_model[key] = dict()
                transition_model[key][next_key] = 1.0

        # Save and log the transition model
        keys = list(keys)
        next_keys = list(next_keys)
        keys.sort()
        next_keys.sort()

        folder = experiment + '/trial_%d' % trial
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(folder + '/transition_model_%d' % (step - 1), 'wb') as f:
            pickle.dump(transition_model, f)

        logger.log("Forward Current States: %s" % ", ".join([str(curr_state) for curr_state in curr_states]))
        for key in sorted(forward_curr_state_corr):
            data_ = forward_curr_state_corr[key]["data"] / forward_curr_state_corr[key]["total"]
            data_ = ", ".join([str(round(d, 2)) for d in data_.tolist()])
            logger.log("%d:                         %s" % (key, data_))
        logger.log("=======================")

        logger.log("Backward Current States: %s" % ", ".join([str(curr_state) for curr_state in curr_states]))
        for key in sorted(backward_curr_state_corr):
            data_ = backward_curr_state_corr[key]["data"] / backward_curr_state_corr[key]["total"]
            data_ = ", ".join([str(round(d, 2)) for d in data_.tolist()])
            logger.log("%d:                         %s" % (key, data_))
        logger.log("=======================")

        logger.log("Forward Next States: %s" % ", ".join([str(next_state) for next_state in next_states]))
        for key in sorted(forward_next_state_corr):
            data_ = forward_next_state_corr[key]["data"] / forward_next_state_corr[key]["total"]
            data_ = ", ".join([str(round(d, 2)) for d in data_.tolist()])
            logger.log("%d:                         %s" % (key, data_))
        logger.log("=======================")

        logger.log("Backward Next States: %s" % ", ".join([str(next_state) for next_state in next_states]))
        for key in sorted(backward_next_state_corr):
            data_ = backward_next_state_corr[key]["data"] / backward_next_state_corr[key]["total"]
            data_ = ", ".join([str(round(d, 2)) for d in data_.tolist()])
            logger.log("%d:                         %s" % (key, data_))
        logger.log("=======================")

        def verify(matrix):
            # n x 3 matrix. Check if each row and each col has a max > 0.95
            col_max = matrix.max(0)
            row_max = matrix.max(1)

            return (col_max > 0.95).all() and (row_max > 0.95).all()

        logger.log("KI Current States: %s" % ", ".join([str(curr_state) for curr_state in curr_states]))
        curr_matrix = []
        for key in sorted(ki_curr_state_corr):
            data_ = ki_curr_state_corr[key]["data"] / ki_curr_state_corr[key]["total"]
            if ki_curr_state_corr[key]["total"] > 25:
                curr_matrix.append(data_)
            data_ = ", ".join([str(round(d, 2)) for d in data_.tolist()])
            logger.log("%r:                         %s" % (key, data_))
        curr_matrix = np.vstack(curr_matrix)
        verify_curr = verify(curr_matrix)
        if not verify_curr:
            logger.log("Automatic check failed for curr of step %d. Did actions match %r" %
                       (step - 1, env.env.opt_a[step - 2] == env.env.opt_b[step - 2]))
        logger.log("=======================")

        logger.log("KI Next States: %s" % ", ".join([str(next_state) for next_state in next_states]))
        next_matrix = []
        for key in sorted(ki_next_state_corr):
            data_ = ki_next_state_corr[key]["data"] / ki_next_state_corr[key]["total"]
            if ki_next_state_corr[key]["total"] > 25:
                next_matrix.append(data_)
            data_ = ", ".join([str(round(d, 2)) for d in data_.tolist()])
            logger.log("%r:                         %s" % (key, data_))
        next_matrix = np.vstack(next_matrix)
        verify_next = verify(next_matrix)
        if not verify_next:
            logger.log("Automatic check failed for next of step %d. Did actions match %r" %
                       (step - 1, env.env.opt_a[step - 1] == env.env.opt_b[step - 1]))
        logger.log("=======================")

        if verify_curr and verify_next:
            logger.log("Automatic check passed for step %d" % (step - 1))

        logger.log("Step %d Model. Matrix of size %d x %d " % (step - 1, len(keys), len(next_keys)))
        logger.log("Special Actions: Step %d (A=%d and B=%d) and Step %d (A=%d and B=%d)" %
                   (step - 2, env.env.opt_a[step - 2], env.env.opt_b[step - 2],
                    step - 1, env.env.opt_a[step - 1], env.env.opt_b[step - 1]))
        logger.log("State and Action |  %s" % ", ".join(next_keys))

        for key in keys:
            count = 0.0
            for next_key in next_keys:
                if next_key in transition_model[key]:
                    count += transition_model[key][next_key]

            if count <= 25:  # Too small to be significant
                continue

            data = ""
            for next_key in next_keys:
                if next_key in transition_model[key]:
                    val = transition_model[key][next_key] / max(1.0, count)
                    data += "%f,  " % val
                else:
                    data += "0.0,  "

            logger.log("%s (%d) -> %s" % (key, count, data))