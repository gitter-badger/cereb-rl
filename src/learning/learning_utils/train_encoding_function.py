import time
import random
import torch.optim as optim

from learning.learning_utils.clustering_algorithm import GreedyClustering, ClusteringModel, FeatureComputation
from model.encoder_model_wrapper import EncoderModelWrapper
from learning.learning_utils.entropy_decay_policy import EntropyDecayPolicy
from learning.learning_utils.train_encoding_function_utils import *


class TrainEncodingFunction:
    """ Class for training the encoding function """

    def __init__(self, config, constants):

        self.config = config
        self.constants = constants
        self.epoch = constants["encoder_training_epoch"]
        self.learning_rate = constants["encoder_training_lr"]
        self.batch_size = constants["encoder_training_batch_size"]
        self.validation_size_portion = constants["validation_data_percent"]
        self.entropy_coeff = constants["entropy_reg_coeff"]
        self.num_homing_policies = constants["num_homing_policy"]

        # Model type with discretization on x' in (x, a, x')
        self.backward_model_type = constants["backward_model_type"]

        # Model type with discretization on x in (x, a, x')
        self.forward_model_type = constants["forward_model_type"]

        self.entropy_decay_policy = EntropyDecayPolicy(constants, self.epoch)
        self.patience = constants["patience"]

        self.max_retrials = constants["max_try"]
        self.expected_optima = constants["expected_optima"]  # If the model reaches this loss then we exit

    def calc_loss(self, model, batch, epoch, discretized, test_set_errors=None, past_entropy=None):

        prev_observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_curr_obs())).view(1, -1)
                                                for point in batch], dim=0)).float()
        actions = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_action())).view(1, -1)
                                      for point in batch], dim=0)).long()
        observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_next_obs())).view(1, -1)
                                           for point in batch], dim=0)).float()
        gold_labels = cuda_var(torch.cat([torch.from_numpy(np.array(point.is_valid())).view(1, -1)
                                          for point in batch], dim=0)).long()

        info_dict = dict()

        # Compute loss
        log_probs, meta_dict = model.gen_log_prob(prev_observations=prev_observations,
                                                  actions=actions,
                                                  observations=observations,
                                                  discretized=discretized)  # outputs a matrix of size batch x 2
        classification_loss = -torch.mean(log_probs.gather(1, gold_labels.view(-1, 1)))

        decay_coeff = self.entropy_decay_policy.get_entropy_coeff(epoch, test_set_errors, past_entropy)

        if discretized:
            # For discretized models, there is an internal classification step representation by a probability
            # distribution that can be controlled using entropy bonus
            loss = classification_loss - self.entropy_coeff * decay_coeff * meta_dict["mean_entropy"]
        else:
            loss = classification_loss

        info_dict["classification_loss"] = classification_loss

        if discretized:
            info_dict["mean_entropy"] = meta_dict["mean_entropy"]
            info_dict["entropy_coeff"] = self.entropy_coeff * decay_coeff
        else:
            info_dict["mean_entropy"] = 0.0
            info_dict["entropy_coeff"] = 0.0

        return loss, info_dict

    def train_model(self, dataset, logger, model_type, bootstrap_model, category, discretized, debug, tensorboard):

        # torch.manual_seed(ctr)

        # Current model
        model = EncoderModelWrapper.get_encoder_model(model_type, self.config, self.constants, bootstrap_model)

        # Model for storing the best model as measured by performance on the test set
        best_model = EncoderModelWrapper.get_encoder_model(model_type, self.config, self.constants, bootstrap_model)

        param_with_grad = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(params=param_with_grad, lr=self.learning_rate)

        random.shuffle(dataset)
        dataset_size = len(dataset)
        batches = [dataset[i:i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]

        train_batch = int((1.0 - self.validation_size_portion) * len(batches))
        train_batches = batches[:train_batch]
        test_batches = batches[train_batch:]

        best_test_loss, best_epoch, train_loss = 0.69, -1, 0.69  # 0.69 is -log(2)
        num_train_examples, num_test_examples = 0, 0
        patience_counter = 0

        test_set_errors, past_entropy = [], []

        for epoch_ in range(1, self.epoch + 1):

            train_loss, mean_entropy, num_train_examples = 0.0, 0.0, 0
            for train_batch in train_batches:

                loss, info_dict = self.calc_loss(model, train_batch, epoch_, discretized, test_set_errors, past_entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                optimizer.step()

                loss = float(loss)
                tensorboard.log_scalar("Encoding_Loss ", loss)

                for key in info_dict:
                    tensorboard.log_scalar(key, info_dict[key])

                batch_size = len(train_batch)
                train_loss = train_loss + float(info_dict["classification_loss"]) * batch_size
                mean_entropy = mean_entropy + float(info_dict["mean_entropy"]) * batch_size
                num_train_examples = num_train_examples + batch_size

            train_loss = train_loss / float(max(1, num_train_examples))
            mean_entropy = mean_entropy / float(max(1, num_train_examples))

            # Evaluate on test batches
            test_loss = 0
            num_test_examples = 0
            for test_batch in test_batches:
                _, info_dict = self.calc_loss(model, test_batch, epoch_, discretized, test_set_errors, past_entropy)

                batch_size = len(test_batch)
                test_loss = test_loss + float(info_dict["classification_loss"]) * batch_size
                num_test_examples = num_test_examples + batch_size

            test_loss = test_loss / float(max(1, num_test_examples))
            logger.debug("Train Loss after epoch %r is %r, mean entropy %r, entropy coeff %r" %
                         (epoch_, round(train_loss, 2), round(mean_entropy, 2), info_dict["entropy_coeff"]))
            logger.debug("Test Loss after epoch %r is %r" % (epoch_, round(test_loss, 2)))

            test_set_errors.append(test_loss)
            past_entropy.append(mean_entropy)

            if test_loss < best_test_loss:
                patience_counter = 0
                best_test_loss = test_loss
                best_epoch = epoch_
                best_model.load_state_dict(model.state_dict())
            else:
                # Check patience condition
                patience_counter += 1  # number of epoch since last increase
                if best_test_loss < self.expected_optima or test_loss > 0.8:  # Found good solution or diverged
                    break

                if patience_counter == self.patience:
                    logger.log("Patience Condition Triggered: No improvement for %r epochs" % patience_counter)
                    break

        logger.log("%s (Discretized: %r), Train/Test = %d/%d, Best Tune Loss %r at epoch %r, "
                   "Train Loss after %r epochs is %r " % (model_type, discretized, num_train_examples,
                                                          num_test_examples, round(best_test_loss, 2),
                                                          best_epoch, epoch_, round(train_loss, 2)))

        if debug and discretized:
            if category == "backward":
                log_model_performance(self.num_homing_policies, best_model, test_batches, best_test_loss, logger)

        return best_model, best_test_loss

    def do_train(self, dataset, logger, tensorboard, debug, bootstrap_model=None, undiscretized_initialization=True,
                 category="backward"):

        # Do not bootstrap if not asked to
        if not self.constants["bootstrap_encoder_model"]:
            bootstrap_model = None

        if self.constants["discretization"]:

            # Train using a discretized model
            encoding_function, _ = self.do_train_with_discretized_models(
                dataset,
                logger,
                tensorboard,
                debug,
                bootstrap_model=bootstrap_model,
                undiscretized_initialization=undiscretized_initialization,
                category=category)

            num_state_budget = self.constants["num_homing_policy"]

        else:
            # Train using an undiscretized model
            encoding_function, result_meta = self.do_train_with_undiscretized_models(dataset,
                                                                                     logger,
                                                                                     tensorboard,
                                                                                     debug,
                                                                                     bootstrap_model=bootstrap_model,
                                                                                     category=category)
            num_state_budget = result_meta["num_clusters"]

        return encoding_function, num_state_budget

    def do_train_with_discretized_models(self, dataset, logger, tensorboard, debug, bootstrap_model=None,
                                         undiscretized_initialization=True, category="backward"):
        """ Given a dataset comprising of (x, a, x', y) where y=1 means the x' was observed on taking action a in x and
            y=0 means it was observed independently of x, a. We train a model to differentiate between the dataset.
            The model we use has a certain structure that enforces discretization. """

        overall_best_model, overall_best_test_loss = None, float('inf')

        if category == "backward":
            model_type = self.backward_model_type
        elif category == "forward":
            model_type = self.forward_model_type
        else:
            raise AssertionError("Unhandled category %s" % category)

        if debug:
            log_dataset_stats(dataset, logger)

        for ctr in range(1, self.max_retrials + 1):

            # torch.manual_seed(ctr)

            # Current model
            if undiscretized_initialization:
                # Learn a undiscretized model
                undiscretized_model, best_test_loss = self.train_model(dataset, logger, model_type, bootstrap_model,
                                                                       category, False, debug, tensorboard)

                # Bootstrap from the learned undiscretized model now
                my_bootstrap_model = undiscretized_model
            else:
                # Bootstrap from the input bootstrap model
                my_bootstrap_model = bootstrap_model

            best_model, best_test_loss = self.train_model(dataset, logger, model_type, my_bootstrap_model,
                                                          category, True, debug, tensorboard)

            if best_test_loss < overall_best_test_loss:
                overall_best_test_loss = best_test_loss
                overall_best_model = best_model

            if overall_best_test_loss < self.expected_optima:
                break
            else:
                logger.log("Failed to reach expected loss. This was attempt number %d" % ctr)

        return overall_best_model, {"loss": overall_best_test_loss,
                                    "success": overall_best_test_loss < self.expected_optima}

    def do_train_with_undiscretized_models(self, dataset, logger, tensorboard, debug, bootstrap_model=None,
                                           category="backward"):

        """ Given a dataset comprising of (x, a, x', y) where y=1 means the x' was observed on taking action a in x
            and y=0 means it was observed independently of x, a. We train a model to differentiate between the dataset
            and perform clustering on the model output to learn a state abstraction function. """

        overall_best_model, overall_best_test_loss = None, float('inf')
        discretized = False

        if category == "backward":
            model_type = self.backward_model_type
        else:
            raise AssertionError("Unhandled category %s" % category)

        for ctr in range(1, self.max_retrials + 1):

            # torch.manual_seed(ctr)

            # Train a undiscretized model on the dataset
            best_model, best_test_loss = self.train_model(dataset, logger, model_type, bootstrap_model,
                                                          category, discretized, debug, tensorboard)

            if best_test_loss < overall_best_test_loss:
                overall_best_test_loss = best_test_loss
                overall_best_model = best_model

            if overall_best_test_loss < self.expected_optima:
                break
            else:
                logger.log("Failed to reach expected loss. This was attempt number %d" % ctr)

        curr_obs_actions = [(dp.curr_obs, dp.action) for dp in dataset if dp.is_valid() == 1]
        valid_dataset = [dp for dp in dataset if dp.is_valid() == 1]

        # Compute features for observations
        timestep_feature_calc_start = time.time()
        logger.debug("Calculating features for clustering steps. Size of dataset: %d" % len(curr_obs_actions))
        feature_fn = FeatureComputation(curr_obs_actions=curr_obs_actions,
                                        model=overall_best_model,
                                        batch_size=1024,
                                        discretized=discretized)
        vectors = [feature_fn.calc_feature(dp_.next_obs) for dp_ in valid_dataset]
        logger.debug("Calculated features. Time taken %d sec" % (time.time() - timestep_feature_calc_start))

        # Call the clustering algorithm to generate clusters
        threshold = 0.0     # TODO use the generalization error to define threshold
        cluster_alg = GreedyClustering(threshold=threshold, dim=feature_fn.dim)
        cluster_centers = cluster_alg.cluster(vectors)
        logger.debug("Number of clusters with L1 distance and threshold %f is %d" % (threshold, len(cluster_centers)))

        # Define the state abstraction model
        encoder_model = ClusteringModel(cluster_centers, feature_fn)

        logger.debug("Mapping datapoints to their center")
        timestep_center_assign_start = time.time()
        for dp_, feature_ in zip(valid_dataset, vectors):
            dp_.meta_dict["cluster_center"] = encoder_model.encode_observations({"vec": feature_})
        logger.debug("Done computing in time %d sec " % (time.time() - timestep_center_assign_start))

        return encoder_model, {"loss": overall_best_test_loss,
                               "success": overall_best_test_loss < self.expected_optima,
                               "num_clusters": len(cluster_centers)}
