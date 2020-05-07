import os
import json
import logging
import argparse
import statistics
import torch.multiprocessing as mp

from environments.environment_package.environment_wrapper import GenerateEnvironmentWrapper
from learning.baselines.ppo.ppo_utils import ppo_feature
from utils.multiprocess_logger import MultiprocessingLoggerManager
from setup_validator.core_validator import validate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='stochcombolock', help="name of the environment e.g., montezuma")
    parser.add_argument("--num_processes", default=6, type=int,
                        help="number of policy search (PS) processes to be launched at a given time")
    parser.add_argument("--forwardmodel", default='forwardmodel', help="Model for training the forwad abstraction")
    parser.add_argument("--backwardmodel", default='backwardmodel', help="Model for learning the backward abstraction")
    parser.add_argument("--discretization", default="True", help="Train with discretized/undiscretized model")
    parser.add_argument("--policy_type", default="linear", type=str, help="Type of policy (linear, non-linear)")
    parser.add_argument("--name", default="neurips", help="Name of the experiment")
    parser.add_argument("--horizon", default="1", type=int, help="Horizon")
    parser.add_argument("--save_path", default="./results/", type=str, help="Folder where to save results")
    args = parser.parse_args()

    env_name = args.env
    num_processes = args.num_processes
    exp_name = args.name

    experiment_name = "ppo-%s-model-%s-horizon-%d-%s" % (exp_name, args.model, args.horizon, env_name)
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

    with open("data/%s/config.json" % env_name) as f:
        config = json.load(f)
        # Add command line arguments. Command line arguments supersede file settings.
        config["horizon"] = args.horizon
        GenerateEnvironmentWrapper.adapt_config_to_domain(env_name, config)
    with open("data/%s/constants.json" % env_name) as f:
        constants = json.load(f)
        constants["model_type"] = args.model
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

    # Create the environment
    env = GenerateEnvironmentWrapper(env_name, config)
    master_logger.log("Environment Created")
    print("Created Environment...")

    homing_policy_validation_fn = env.generate_homing_policy_validation_fn()

    performance = []
    for attempt in range(1, 6):
        master_logger.log("========= STARTING EXPERIMENT %d ======== " % attempt)

        num_samples_half_regret = ppo_feature(experiment, env, config, constants, master_logger,
                                              use_pushover=False, debug=False)
        performance.append(num_samples_half_regret)
    master_logger.log("Median Performance %r. All performance %r" % (statistics.median(performance),
                                                                     performance))
    print("All performance ", performance)
    print("Median performance ",statistics.median(performance))

if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
