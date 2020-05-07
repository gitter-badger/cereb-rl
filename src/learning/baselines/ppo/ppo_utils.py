import time

from learning.baselines.ppo.config import Config


def run_steps(agent, experiment, env, config, constants, logger, use_pushover, debug):
    """ Execute the agent """

    ppo_config = agent.ppo_config
    t0 = time.time()

    cumm_total_reward = 0.0
    optimal_total_reward = 0.0
    step = 0
    optimal_v = env.get_optimal_value()
    regret_ratio = 0.0

    while step <= ppo_config.max_steps:

        if ppo_config.log_interval and not step % ppo_config.log_interval:
            logger.log("steps %d, %.2f steps/s. Regret ratio %r" %
                       (step, ppo_config.log_interval / (time.time() - t0), regret_ratio))
            t0 = time.time()

        # Collect some samples and do an update
        total_reward = agent.step(env, ppo_config.num_episodes_per_update, logger)

        cumm_total_reward += total_reward
        optimal_total_reward += optimal_v * ppo_config.num_episodes_per_update
        step += ppo_config.num_episodes_per_update

        regret_ratio = cumm_total_reward / float(max(1.0, optimal_total_reward))

        if regret_ratio >= 0.5:
            print("Reached half regret after %r steps." % step)
            logger.log("Reached half regret after %r steps." % step)
            return step

    print("Maximum step exceeded")
    logger.log("Maximum step exceeded")

    return float('inf')


def ppo_feature(experiment, env, config, constants, logger, use_pushover, debug):
    """ Run the PPO Baseline """

    ppo_config = Config()
    ppo_config.num_episodes_per_update = 10
    ppo_config.discount = 0.99
    ppo_config.use_gae = True
    ppo_config.gae_tau = 0.95
    ppo_config.entropy_weight = 0.01
    ppo_config.gradient_clip = 5
    ppo_config.rollout_length = config["horizon"]  # TODO CHECK
    ppo_config.optimization_epochs = 10
    ppo_config.mini_batch_size = 32
    ppo_config.ppo_ratio_clip = 0.2
    ppo_config.log_interval = 10 * ppo_config.num_episodes_per_update
    ppo_config.max_steps = 1000000  # This means we let the agent run for 10M episodes

    agent = PPOAgent(ppo_config, config, constants)

    run_steps(agent, experiment, env, config, constants, logger, use_pushover, debug)
