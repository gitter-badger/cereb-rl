import os
import json
import logging
import argparse
import torch.multiprocessing as mp

from environments.environment_package.environment_wrapper import GenerateEnvironmentWrapper
from utils.multiprocess_logger import MultiprocessingLoggerManager
from setup_validator.core_validator import validate
from learning.learning_utils.debug_train_encoding_function import DebugTrainEncodingFunction


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='stochcombolock', help="name of the environment e.g., montezuma")
    parser.add_argument("--model", default='gumbeldouble', help="Model for training the encoding function")
    parser.add_argument("--name", default="debug-encoder", help="Name of the experiment")
    parser.add_argument("--horizon", default=-1, type=int, help="Horizon")
    parser.add_argument("--samples", default=-1, type=int, help="Samples")
    parser.add_argument("--learn_type", default="vanilla", type=str, help="Either vanilla, coordinate, transfer")
    args = parser.parse_args()

    env_name = args.env
    exp_name = args.name

    with open("data/%s/config.json" % env_name) as f:
        config = json.load(f)
        # Add command line arguments. Command line arguments supersede file settings.
        if args.horizon != -1:
            config["horizon"] = args.horizon
        config["encoder_training_type"] = args.learn_type
        GenerateEnvironmentWrapper.adapt_config_to_domain(env_name, config)
    with open("data/%s/constants.json" % env_name) as f:
        constants = json.load(f)
        if args.samples != -1:
            constants["encoder_training_num_samples"] = args.samples
        constants["model_type"] = args.model
    print(json.dumps(config, indent=2))

    # Validate the keys
    validate(config, constants)

    # Create file
    experiment_name = "%s-model-%s-horizon-%d-samples-%d-%s" % (exp_name, args.model, config["horizon"],
                                                                constants["encoder_training_num_samples"], env_name)
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

    # performance = []
    num_runs = 100
    for attempt in range(1, num_runs + 1):
        master_logger.log("========= STARTING EXPERIMENT %d ======== " % attempt)

        p = mp.Process(target=DebugTrainEncodingFunction.do_train, args=(config, constants, env_name,
                                                                         experiment_name, master_logger, False, True))
        p.daemon = False
        p.start()
        p.join()

    # for key in performance[0]:  # Assumes the keys are same across all runes
    #     results = [result[key] for result in performance]
    #     master_logger.log("%r: Mean %r, Median %r, Std %r, Num runs %r, All performance %r" %
    #                       (key, statistics.mean(results), statistics.median(results), statistics.stdev(results),
    #                        num_runs, results))
    #     print("%r: Mean %r, Median %r, Std %r, Num runs %r, All performance %r" %
    #                       (key, statistics.mean(results), statistics.median(results), statistics.stdev(results),
    #                        num_runs, results))

    # Cleanup
    multiprocess_logging_manager.cleanup()

if __name__ == "__main__":

    print("SETTING THE START METHOD ")
    mp.freeze_support()
    mp.set_start_method('spawn')
    main()
