import sys
import argparse
import signal
import os
import copy

import environment_creator
from pdqfd_simple import SimplePDQFDLearner
from q_network_simple import SimpleQNetwork
from misc_utils import boolean_flag

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def main(args):
    logger.debug('Configuration: {}'.format(args))

    network_creator, env_creator = get_network_and_environment_creator(args)

    learner = SimplePDQFDLearner(network_creator, env_creator, args)

    setup_kill_signal_handler(learner)

    logger.info('Starting training')
    learner.train()
    logger.info('Finished training')


def setup_kill_signal_handler(learner):
    main_process_pid = os.getpid()

    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logger.info('Signal ' + str(signal) + ' detected, cleaning up.')
            learner.cleanup()
            logger.info('Cleanup completed, shutting down...')
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def get_network_and_environment_creator(args, random_seed=3):
    args.random_seed = random_seed
    env_creator = environment_creator.EnvironmentCreator(args)
    num_actions = env_creator.num_actions
    args.num_actions = num_actions
    print(env_creator.state_shape)
    network_conf = {'num_actions': num_actions,
                    'expert_margin': args.expert_margin,
                    'margin_loss_coeff': args.margin_loss_coeff,
                    'L2_reg_coeff': args.L2_reg_coeff,
                    'device': args.device,
                    'clip_loss_delta': args.clip_loss_delta,
                    'clip_norm': args.clip_norm,
                    'clip_norm_type': args.clip_norm_type,
                    'arch': args.arch,
                    'mlp_hiddens': args.mlp_hiddens,
                    'layer_norm': args.layer_norm,
                    'input_shape': env_creator.state_shape,
                    'target_update_tau': args.target_update_tau,
                    'continuous_target_update': args.continuous_target_update}
    
    network = SimpleQNetwork

    def network_creator(name='value_learning', learning_network=None):
        nonlocal network_conf
        copied_network_conf = copy.copy(network_conf)
        copied_network_conf['name'] = name
        return network(copied_network_conf, learning_network=learning_network)

    return network_creator, env_creator


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default='CorridorFLNonSkid-v1', help='Name of game', dest='game')
    parser.add_argument('-d', '--device', default='/cpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    parser.add_argument('--rom_path', default='./atari_roms', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer to be used: Adam or Rmsprop")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float, help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    #parser.add_argument('--entropy', default=0.02, type=float, help="Strength of the entropy regularization term (needed for actor-critic)", dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=3.0, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('--max_local_steps', default=5, type=int, help="Number of steps to gain experience from before every update.", dest="max_local_steps")
    parser.add_argument('--arch', default='mlp', help="Which network architecture to use: MLP, Deepmind's NIPS or NATURE", dest="arch")
    parser.add_argument('--mlp_hiddens', default='[50, 50]',
                        help="Hidden layers for MLP", dest="mlp_hiddens")
    boolean_flag(parser, "layer_norm", default=False, help="whether or not to use layer normalization")

    parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")

    # Prioritized experience replay
    boolean_flag(parser, "use_exp_replay", default=True, help="whether or not to use experience replay")
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6), help="replay buffer size", dest="replay_buffer_size")
    boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
    parser.add_argument("--prioritized_alpha", type=float, default=0.4, help="alpha parameter for prioritized replay buffer", dest="prioritized_alpha")
    parser.add_argument("--prioritized_beta0", type=float, default=0.6, help="initial value of beta parameters for prioritized replay", dest="prioritized_beta0")
    parser.add_argument("--prioritized_eps", type=float, default=1e-3, help="eps parameter for prioritized replay buffer", dest="prioritized_eps")

    # PDQfD arguments 
    boolean_flag(parser, "demo", default=True, help="whether or not to use demonstration data")
    parser.add_argument("--demo_db", type=str, default=None, help="path to hdf5 file containing demo data")
    parser.add_argument("--demo_trans_size", type=int, default=int(10500), help="number of demo transitions")
    parser.add_argument("--demo_model_dir", type=str, default=None, help="load demonstration model from this directory")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    parser.add_argument("--pre_train_steps", type=int, default=int(750000), help="number of steps to learn from demo transitions alone")
    parser.add_argument('--margin_loss_coeff', default=1.0, type=float, help="margin loss coefficient", dest="margin_loss_coeff")
    parser.add_argument('--L2_reg_coeff', default=1e-5, type=float, help="l2 regularisation coefficient", dest="L2_reg_coeff")
    #parser.add_argument("--n_step_loss_coeff", type=float, default=1.0, help="n-step loss coefficient")
    parser.add_argument("--expert_margin", type=float, default=0.8, help="margin with which expert action values to be above other values", dest="expert_margin")
    parser.add_argument("--prioritized_eps_d", type=float, default=1.0, help="eps parameter for demo transitions in prioritized replay buffer", dest="prioritized_eps_d")
    #parser.add_argument("--n_step", type=int, default=int(10), help="number of steps agent to look ahead for returns")
    parser.add_argument('--exp_epsilon', default=1, type=float, help="Epsilon for epsilon greedy exploration", dest="exp_epsilon")
    parser.add_argument('--alg_type', default='value', help="Class of RL algorithms -- value, policy", dest="alg_type")
    parser.add_argument('--clip_loss', default=1.0, type=float, help="Delta for Huber loss. Default = 1.0", dest="clip_loss_delta")
    boolean_flag(parser, "continuous_target_update", default=True, help="Whether to update target network at fixed intervals or progressively")
    parser.add_argument("--target_update_freq", type=int, default=10000, help="number of steps between every target network update", dest="target_update_freq")
    parser.add_argument("--target_update_tau", type=float, default=0.001,
                        help="tau for csoft, continuous target netwok update: q_target_param = tau*q_learning_param + (1-tau)*q_target_param", dest="target_update_tau")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of transitions/steps to read from the exp. replay and train with")
    # Extra argument -- not in original DQfD
    parser.add_argument('--demo_train_ratio', default=0.2, type=float, help="Demo/Emulator training ratio", dest="demo_train_ratio")
    boolean_flag(parser, "double_q", default=True, help="Whether or not to use Double Q-learning")

    # Demo agent
    parser.add_argument('-f', '--demo_agent_folder', type=str, help="Folder where demo agent is stored.", dest="demo_agent_folder")
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-se', '--serial_episodes', default='10', type=int, help="Number of serial episodes", dest="serial_episodes")

    parser.add_argument('--experiment_type', default='corridor', type=str, help="Class of environments to experiment with, e.g. atari, corridor, etc.", dest="experiment_type")
    boolean_flag(parser, "debug", default=False, help="Whether or not to print debug messages")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    import logger_utils
    logger_utils.save_args(args, args.debugging_folder)

    main(args)
