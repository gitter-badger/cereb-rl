#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


from deep_rl import *
import argparse, os


# PPO
def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 100
    config.log_interval = 1000
    if config.normalize_bonus_rewards == 1:
        config.reward_bonus_normalizer = MeanStdNormalizer()
    else:
        config.reward_bonus_normalizer = RescaleNormalizer()
        
    if config.system == 'gcr':
        config.log_dir = './log/'
    elif config.system == 'philly':
        config.log_dir = os.getenv('PT_OUTPUT_DIR') + '/'

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, single_process=True, seed=config.seed, horizon = config.horizon)
    config.eval_env = Task(config.game, seed=config.seed, horizon=config.horizon)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.lr)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, FCBody(config.state_dim))

    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = config.horizon
    config.optimization_epochs = 10
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.max_steps = 10e6*config.horizon
    run_steps(PPOAgent(config))



if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    random_seed()
    select_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-alg', type=str, default='ppo')
    parser.add_argument('-bonus_type', type=str, default='oracle-cnts')
    parser.add_argument('-env', type=str, default='diabcombolock')
    parser.add_argument('-horizon', type=int, default=3)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-entropy_weight', type=float, default=0.01)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-bonus_coeff', type=float, default=100)
    parser.add_argument('-normalize_bonus_rewards', type=int, default=0)
    parser.add_argument('-system', type=str, default='gcr')
    config = parser.parse_args()
    select_device(config.device)

    if config.alg == 'ppo':
        ppo_feature(game=config.env,
                    lr=config.lr,
                    horizon=config.horizon,
                    seed=config.seed,
                    entropy_weight = config.entropy_weight,
                    bonus_type = 'none',
                    alg='ppo',
                    normalize_bonus_rewards=0,
                    system = config.system)        
    elif config.alg == 'ppo-rnd':
        ppo_feature(game=config.env,
                    lr=config.lr,
                    horizon=config.horizon,
                    seed=config.seed,
                    entropy_weight = config.entropy_weight,
                    bonus_type = 'rnd',
                    bonus_coeff = config.bonus_coeff,
                    alg='ppo-rnd',
                    normalize_bonus_rewards=config.normalize_bonus_rewards,
                    system = config.system)
    elif config.alg == 'ppo-oracle-cnts':
        ppo_feature(game=config.env,
                    lr=config.lr,
                    horizon=config.horizon,
                    seed=config.seed,
                    entropy_weight = config.entropy_weight,
                    bonus_type = 'oracle-cnts',
                    bonus_coeff = config.bonus_coeff,
                    alg='ppo-oracle-cnts',
                    normalize_bonus_rewards=config.normalize_bonus_rewards,
                    system = config.system)
