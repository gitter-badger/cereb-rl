import os
import torch

from src.learning.baselines.du_baselines import Params, OracleQ, Decoding, QLearning

torch.set_default_tensor_type(torch.DoubleTensor)


def get_alg(args, config):

    name = args.alg
    assert type(config["obs_dim"]) == int, "Can only handle 1D features"

    if name == "oracleq":
        alg_dict = {'horizon': args.horizon,
                    'alpha': args.lr,
                    'conf': args.conf}
        alg = OracleQ.OracleQ(config["num_actions"], params=alg_dict)

    elif name == 'decoding':
        alg_dict = {'horizon': config["horizon"],
                    'model_type': args.model_type,
                    'n': args.n,
                    'num_cluster': args.num_cluster}
        alg = Decoding.Decoding(config["obs_dim"], config["num_actions"], params=alg_dict)

    elif name == 'qlearning':
        assert args.tabular, "[EXPERIMENT] Must run QLearning in tabular mode"
        alg_dict = {
            'alpha': float(args.lr),
            'epsfrac': float(args.epsfrac),
            'num_episodes': int(args.episodes)}
        alg = QLearning.QLearning(config["num_actions"], params=alg_dict)

    else:
        raise AssertionError("Unhandled case. name is %r" % name)

    return alg


def train(env, alg, args, logger):

    P = Params.Params(vars(args))
    fname = P.get_output_file_name()

    if os.path.isfile(fname):
        logger.log("[EXPERIMENT] Already completed")
        return None

    T = args.episodes
    running_reward = 0.0
    running_opt_reward = 0.0

    optimal_policy_v = env.get_optimal_value()

    if optimal_policy_v is None:
        optimal_policy_v = float('inf')   # This way you can never compete so the algorithm runs for max no. of episodes

    total_episodes = 0

    for t in range(1, T + 1):

        state, _ = env.reset()
        done = False

        while not done:
            action = alg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            alg.save_transition(state, action, reward, next_state)
            state = next_state
            running_reward += reward

        alg.finish_episode()

        running_opt_reward += optimal_policy_v
        ratio = running_reward / max(1.0, running_opt_reward)
        estimated_value_function = running_reward / float(t)

        if t % 1000 == 0:
            logger.log("[EXPERIMENT] Episode %d Completed. Average reward: %0.2f, Ratio %r" %
                       (t, estimated_value_function, ratio))

        if running_reward >= 0.5 * running_opt_reward:
            total_episodes = t
            logger.log("Exceeded half of V^* after %r episodes. Estimate V %r" % (t, estimated_value_function))
            break

    return {"total_episodes": total_episodes}
