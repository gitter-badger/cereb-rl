import torch

from utils.cuda import cuda_var


def evaluate(env, policy, horizon, logger, train_episodes, sum_train_reward):
    """ Compute mean total reward and the number of episodes to reach half regret of the optimal policy """

    optimal_policy_v = env.get_optimal_value()

    if optimal_policy_v is None:
        # Evaluate the policy based on fixed number of episodes and computing total reward
        return evaluate_for_policy_value(env, policy, horizon, logger, train_episodes)
    else:
        # Evaluate the policy based on total number of episodes needed to reach half of optimal regret
        return evaluate_for_half_regret(env, policy, horizon, optimal_policy_v,
                                        logger, train_episodes, sum_train_reward)


def generate_failure_result(env, train_samples):
    """ When the agent fails in the middle of training. This function allows it to return the intermediate result """

    optimal_policy_v = env.get_optimal_value()
    if optimal_policy_v is None:

        return {"total_train_episodes": train_samples,
                "total_test_episodes": 0,
                "mean_total_test_reward": float('-inf')}

    else:
        return {"total_episodes_half_regret": float('inf'),
                "total_train_episodes": train_samples,
                "total_test_episodes": 0,
                "mean_total_test_reward": float('-inf')}


def evaluate_for_policy_value(env, policy, horizon, logger, train_episodes, num_eval_episodes=100):
    """ Evaluate the policy based on estimate of value function """

    learned_policy_cumm = 0.0

    for _ in range(1, num_eval_episodes + 1):

        # Rollin for steps
        obs, meta = env.reset()
        total_reward = 0.0

        for step in range(1, horizon + 1):
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
            action = policy[step].sample_action(obs_var)
            obs, reward, done, meta = env.step(action)
            total_reward += reward

        learned_policy_cumm += total_reward

    logger.log("Policy mean reward %r" % (learned_policy_cumm / max(1.0, float(num_eval_episodes))))

    return {"total_train_episodes": train_episodes,
            "total_test_episodes": num_eval_episodes,
            "mean_total_test_reward": learned_policy_cumm / float(max(1, num_eval_episodes))}


def evaluate_for_half_regret(env, policy, horizon, optimal_policy_v, logger,
                             train_episodes, sum_train_reward, max_episodes=10000000):
    """ Evaluate the policy based on number of samples needed to reach half the regret """

    optimal_policy_cumm = optimal_policy_v * train_episodes
    learned_policy_cumm = sum_train_reward
    test_episode = 0
    logger.log("Evaluating Policy. V* = %r, At init (num train episodes %d): Total reward %r (Optimal) vs %r (System)" %
               (optimal_policy_v, train_episodes, optimal_policy_cumm, learned_policy_cumm))

    for test_episode in range(1, max_episodes + 1):

        # Rollin for steps
        start_obs, meta = env.reset()
        obs = start_obs
        total_reward = 0.0

        for step in range(1, horizon + 1):
            obs_var = cuda_var(torch.from_numpy(obs)).float().view(1, -1)
            action = policy[step].sample_action(obs_var)
            obs, reward, done, meta = env.step(action)
            total_reward += reward

        optimal_policy_cumm += optimal_policy_v
        learned_policy_cumm += total_reward
        ratio = learned_policy_cumm / float(max(1.0, optimal_policy_cumm))
        estimated_value_function = learned_policy_cumm / float(train_episodes + test_episode)

        if test_episode % 10000 == 0:
            logger.log("(Total Episodes %d) Total reward %r (Optimal) vs %r (System). Ratio %r" % (
                train_episodes + test_episode,  optimal_policy_cumm, learned_policy_cumm, ratio))

        if learned_policy_cumm >= 0.5 * optimal_policy_cumm:
            logger.log("Exceeded half of V^* after %r test episodes. Total test+train episodes %r. Estimate V %r" %
                       (test_episode, train_episodes + test_episode, estimated_value_function))
            print("Exceeded half of V^* after %r test episodes. Total test+train episodes %r. Estimate V %r" %
                  (test_episode, train_episodes + test_episode, estimated_value_function))

            result = {"total_episodes_half_regret": train_episodes + test_episode,
                      "total_train_episodes": train_episodes,
                      "total_test_episodes": test_episode,
                      "mean_total_test_reward": estimated_value_function}

            return result

    logger.log("Exceeded max steps. Learned/V* ratio is %r" %
               (learned_policy_cumm / max(0.00001, float(optimal_policy_cumm))))
    print("Exceeded max steps. Learned/V* ratio is %r" % (
        learned_policy_cumm / max(0.00001, float(optimal_policy_cumm))))

    result = {"total_episodes_half_regret": float('inf'),
              "total_train_episodes": train_episodes,
              "total_test_episodes": test_episode,
              "mean_total_test_reward": learned_policy_cumm / float(train_episodes + test_episode)}

    return result
