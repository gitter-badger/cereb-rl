import os
import json
import random
import torch
import logging
import argparse
import statistics
import numpy as np
import torch.multiprocessing as mp
import learning.baselines.du_baselines.Experiment as du_baseline

from environments.environment_package.environment_wrapper import GenerateEnvironmentWrapper
from utils.multiprocess_logger import MultiprocessingLoggerManager
from setup_validator.core_validator import validate


def main():

    parser = argparse.ArgumentParser(description='du_baselines Experiments')

    parser.add_argument("--env", default='diabcombolock', help="name of the environment e.g., montezuma")
    parser.add_argument("--name", default="run-du-baselines", help="Name of the experiment")
    parser.add_argument("--horizon", default=-1, type=int, help="Horizon")
    parser.add_argument("--noise", default=None, type=str, help="Noise")
    parser.add_argument("--save_trace", default="False", help="Save traces")
    parser.add_argument("--trace_sample_rate", default=500, type=int, help="How often to save traces")
    parser.add_argument("--save_path", default="./results/", type=str, help="Folder where to save results")
    parser.add_argument("--debug", default="False", help="Debug the run")
    parser.add_argument("--pushover", default="False", help="Use pushover to send results on phone")

    # Options for Du Baselines
    parser.add_argument('--seed', type=int, default=367, metavar='N', help='random seed (default: 367)')
    parser.add_argument('--episodes', type=int, default=10000000, help='Training Episodes')
    parser.add_argument('--alg', type=str, default='decoding',
                        help='Learning Algorithm', choices=["oracleq", "decoding", "qlearning"])
    parser.add_argument('--model_type', type=str, default='linear',
                        help='What model class for function approximation', choices=['linear', 'nn'])
    parser.add_argument('--lr', type=float,
                        help='Learning Rate for optimization-based algorithms', default=3e-2)
    parser.add_argument('--epsfrac', type=float,
                        help='Exploration fraction for Baseline DQN.', default=0.1)
    parser.add_argument('--conf', type=float,
                        help='Exploration Bonus Parameter for Oracle Q.', default=3e-2)
    parser.add_argument('--n', type=int, default=200,
                        help="Data collection parameter for decoding algoithm.")
    parser.add_argument('--num_cluster', type=int, default=3,
                        help="Num of hidden state parameter for decoding algoithm.")

    args = parser.parse_args()

    env_name = args.env
    exp_name = args.name

    experiment_name = "%s-%s-model-%s-horizon-%d-samples-%d-noise-%s" % \
                      (exp_name, env_name, args.model_type, args.horizon, args.episodes, args.noise)
    experiment = "./%s/%s" % (args.save_path, experiment_name)
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder
    if not os.path.exists(experiment):
        os.makedirs(experiment)

    # Define log settings
    log_path = experiment + '/train_homer.log'
    multiprocess_logging_manager = MultiprocessingLoggerManager(
        file_path=log_path, logging_level=logging.INFO)
    master_logger = multiprocess_logging_manager.get_logger("Master")
    master_logger.log("----------------------------------------------------------------")
    master_logger.log("                    STARING NEW EXPERIMENT                      ")
    master_logger.log("----------------------------------------------------------------")
    master_logger.log("Environment Name %r. Experiment Name %r" % (env_name, exp_name))

    # Read configuration and constant files. Configuration contain environment information and
    # constant file contains hyperparameters for the model and learning algorithm.
    with open("data/%s/config.json" % env_name) as f:
        config = json.load(f)
        # Add command line arguments. Command line arguments supersede file settings.
        if args.horizon != -1:
            config["horizon"] = args.horizon
        if args.noise is not None:
            config["noise"] = args.noise

        config["save_trace"] = args.save_trace == "True"
        config["trace_sample_rate"] = args.trace_sample_rate
        config["save_path"] = args.save_path
        config["exp_name"] = experiment_name

        GenerateEnvironmentWrapper.adapt_config_to_domain(env_name, config)
    with open("data/%s/constants.json" % env_name) as f:
        constants = json.load(f)
    print(json.dumps(config, indent=2))

    # Validate the keys
    validate(config, constants)

    # log core experiment details
    master_logger.log("CONFIG DETAILS")
    for k, v in sorted(config.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("CONSTANTS DETAILS")
    for k, v in sorted(constants.items()):
        master_logger.log("    %s --- %r" % (k, v))
    master_logger.log("START SCRIPT CONTENTS")
    with open(__file__) as f:
        for line in f.readlines():
            master_logger.log(">>> " + line.strip())
    master_logger.log("END SCRIPT CONTENTS")

    performance = []
    num_runs = 5
    for trial in range(1, num_runs + 1):

        master_logger.log("========= STARTING EXPERIMENT %d ======== " % trial)

        random.seed(args.seed + trial * 29)
        np.random.seed(args.seed + trial * 29)
        torch.manual_seed(args.seed + trial * 37)

        # Create a new environment
        env = GenerateEnvironmentWrapper(env_name, config)
        master_logger.log("Environment Created")
        print("Created Environment...")

        # Save the environment for reproducibility
        env.save_environment(experiment, trial_name=trial)
        print("Saving Environment...")

        learning_alg = du_baseline.get_alg(args, config)
        policy_result = du_baseline.train(env, learning_alg, args, master_logger)

        performance.append(policy_result)

    for key in performance[0]:  # Assumes the keys are same across all runes
        results = [result[key] for result in performance]
        master_logger.log("%r: Mean %r, Median %r, Std %r, Num runs %r, All performance %r" %
                          (key, statistics.mean(results), statistics.median(results), statistics.stdev(results),
                           num_runs, results))
        print("%r: Mean %r, Median %r, Std %r, Num runs %r, All performance %r" %
              (key, statistics.mean(results), statistics.median(results), statistics.stdev(results), num_runs, results))

    # Cleanup
    multiprocess_logging_manager.cleanup()


if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
