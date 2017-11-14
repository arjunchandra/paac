import numpy as np
from multiprocessing import Process
import tensorflow as tf
from logger_utils import variable_summaries
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


CHECKPOINT_INTERVAL = 1000000
 

class ActorLearner(Process):
    
    def __init__(self, network_creator, environment_creator, args):
        
        super(ActorLearner, self).__init__()

        tf.reset_default_graph()

        self.workers = args.emulator_workers

        self.global_step = 0

        self.environment_creator = environment_creator

        self.max_local_steps = args.max_local_steps
        self.state_shape = environment_creator.state_shape
        self.num_actions = args.num_actions
        self.initial_lr = args.initial_lr
        self.lr_annealing_steps = args.lr_annealing_steps
        self.emulator_counts = args.emulator_counts
        self.device = args.device
        self.debugging_folder = args.debugging_folder
        self.network_checkpoint_folder = os.path.join(self.debugging_folder, 'checkpoints/')
        self.optimizer_checkpoint_folder = os.path.join(self.debugging_folder, 'optimizer_checkpoints/')
        self.last_saving_step = 0
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.debugging_folder, 'tf'))

        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer_variable_names = 'OptimizerVariables'
        if args.optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, name=optimizer_variable_names)
        else:
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=args.alpha, epsilon=args.e,
                                                   name=optimizer_variable_names)

        self.emulators = np.asarray([environment_creator.create_environment(i)
                                     for i in range(self.emulator_counts)])
        self.max_global_steps = args.max_global_steps
        self.gamma = args.gamma
        self.game = args.game

        self.network = network_creator(name='value_learning')
        self.target_network = network_creator(name='value_target', learning_network=self.network)
        self.target_update_freq = args.target_update_freq

        # Optimizer
        grads_and_vars = self.optimizer.compute_gradients(self.network.loss, self.network.params)

        self.flat_raw_gradients = tf.concat([tf.reshape(g, [-1]) for g, v in grads_and_vars], axis=0)

        # This is not really an operation, but a list of gradient Tensors.
        # When calling run() on it, the value of those Tensors
        # (i.e., of the gradients) will be calculated
        if args.clip_norm_type == 'ignore':
            # Unclipped gradients
            global_norm = tf.global_norm([g for g, v in grads_and_vars], name='global_norm')
        elif args.clip_norm_type == 'global':
            # Clip network grads by network norm
            gradients_n_norm = tf.clip_by_global_norm(
                [g for g, v in grads_and_vars], args.clip_norm)
            global_norm = tf.identity(gradients_n_norm[1], name='global_norm')
            grads_and_vars = list(zip(gradients_n_norm[0], [v for g, v in grads_and_vars]))
        elif args.clip_norm_type == 'local':
            # Clip layer grads by layer norm
            gradients = [tf.clip_by_norm(
                g, args.clip_norm) for g in grads_and_vars]
            grads_and_vars = list(zip(gradients, [v for g, v in grads_and_vars]))
            global_norm = tf.global_norm([g for g, v in grads_and_vars], name='global_norm')
        else:
            raise Exception('Norm type not recognized')
        self.flat_clipped_gradients = tf.concat([tf.reshape(g, [-1]) for g, v in grads_and_vars], axis=0)

        self.train_step = self.optimizer.apply_gradients(grads_and_vars)

        config = tf.ConfigProto()
        if 'gpu' in self.device:
            logger.debug('Dynamic gpu mem allocation')
            config.gpu_options.allow_growth = True

        self.session = tf.Session(config=config)

        self.network_saver = tf.train.Saver()

        # Summaries
        variable_summaries(self.flat_raw_gradients, 'raw_gradients')
        variable_summaries(self.flat_clipped_gradients, 'clipped_gradients')
        tf.summary.scalar('global_norm', global_norm)
        tf.summary.scalar("TD_loss", self.network.td_loss)

    def save_vars(self, force=False):
        if force or self.global_step - self.last_saving_step >= CHECKPOINT_INTERVAL:
            self.last_saving_step = self.global_step
            self.network_saver.save(self.session, self.network_checkpoint_folder, global_step=self.last_saving_step)


    def rescale_reward(self, reward, type='log'):
        if type == 'log':
            reward = np.sign(reward) * np.log(1 + np.abs(reward))
        elif type == 'normalize':
            reward = 1.0 * reward / self.max_reward
        else:
            """ Clip immediate reward """
            if reward > 1.0:
                reward = 1.0
            elif reward < -1.0:
                reward = -1.0
        return reward

    def init_network(self):
        import os
        if not os.path.exists(self.network_checkpoint_folder):
            os.makedirs(self.network_checkpoint_folder)
        if not os.path.exists(self.optimizer_checkpoint_folder):
            os.makedirs(self.optimizer_checkpoint_folder)

        # Since we only save and restore network trainable variables and optimizer variables, we always need to
        # initialize the other variables
        self.session.run(tf.global_variables_initializer())

        # This should restore both the local/learning network and the target network
        last_saving_step = self.network.init(self.network_checkpoint_folder, self.network_saver, self.session)

        return last_saving_step

    def get_lr(self):
        if self.global_step <= self.lr_annealing_steps:
            return self.initial_lr - (self.global_step * self.initial_lr / self.lr_annealing_steps)
        else:
            return 0.0

    def cleanup(self):
        self.save_vars(True)
        self.session.close()

