import os
import json
import logging
import argparse
import statistics
import torch.multiprocessing as mp

from os import listdir
from os.path import isdir, join
from environments.environment_package.environment_wrapper import GenerateEnvironmentWrapper
from learning.core_learner.homer import DistributedHomerAlgorithm
from utils.multiprocess_logger import MultiprocessingLoggerManager
from setup_validator.core_validator import validate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='diabcombolock', help="name of the environment e.g., montezuma")
    parser.add_argument("--name", default="learn-model", help="Name of the experiment")
    parser.add_argument("--forwardmodel", default='forwardmodel', help="Model for training the forwad abstraction")
    parser.add_argument("--backwardmodel", default='backwardmodel', help="Model for learning the backward abstraction")
    parser.add_argument("--discretization", default="True", help="Train with discretized/undiscretized model")
    parser.add_argument("--policy_type", default="linear", type=str, help="Type of policy (linear, non-linear)")
    parser.add_argument("--load", help="Name of the result folder containing homing policies and environment")
    parser.add_argument("--noise", default=None, type=str, help="Noise")
    parser.add_argument("--samples", default=-1, type=int, help="Samples")
    parser.add_argument("--save_path", default="./results/", type=str, help="Folder where to save results")
    parser.add_argument("--trace_sample_rate", default=500, type=int, help="How often to save traces")
    parser.add_argument("--trace_folder", default="./saved_traces", type=str, help="Folder where to save traces")

    args = parser.parse_args()

    env_name = args.env
    exp_name = args.name
    load_folder = args.load

    experiment_name = "%s-%s-model-%s-noise-%s" % (exp_name, env_name, args.model, args.noise)
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
        if args.noise is not None:
            config["noise"] = args.noise

        config["env_seed"] = args.env_seed
        config["save_trace"] = args.save_trace == "True"
        config["trace_sample_rate"] = args.trace_sample_rate
        config["save_path"] = args.save_path
        config["exp_name"] = experiment_name
        config["policy_type"] = args.policy_type

        GenerateEnvironmentWrapper.adapt_config_to_domain(env_name, config)
    with open("data/%s/constants.json" % env_name) as f:
        constants = json.load(f)
        constants["model_type"] = args.model
        if args.samples != -1:
            constants["encoder_training_num_samples"] = args.samples
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

    num_runs = 5
    for trial in range(1, num_runs + 1):

        master_logger.log("========= STARTING EXPERIMENT %d ======== " % trial)

        # Create a new environment
        print("Created Environment...")
        env = GenerateEnvironmentWrapper(env_name, config)
        master_logger.log("Environment Created")

        # Load the environment
        env_folder = load_folder + "/trial_%d_env" % trial
        env_folders = [join(env_folder, f) for f in listdir(env_folder) if isdir(join(env_folder, f))]
        assert len(env_folders) == 1, "Found more than environment. Specify the folder manually %r" % env_folders
        env.load_environment_from_folder(env_folders[0])
        master_logger.log("Loaded Environment from %r" % env_folders[0])

        # Fix config to match the env.
        # TODO implement the next block of code in a scalable manner
        config["horizon"] = env.env.horizon
        config["obs_dim"] = -1
        GenerateEnvironmentWrapper.adapt_config_to_domain(env_name, config)
        master_logger.log("Environment horizon %r, Observation dimension %r" % (config["horizon"], config["obs_dim"]))

        learning_alg = DistributedHomerAlgorithm(config, constants)

        learning_alg.learn_model(env=env,
                                 load_folder=load_folder,
                                 experiment_name=experiment_name,
                                 logger=master_logger,
                                 use_pushover=False,
                                 experiment=experiment,
                                 trial=trial)

    # Cleanup
    multiprocess_logging_manager.cleanup()


if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
