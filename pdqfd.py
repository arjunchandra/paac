import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging

from emulator_runner import EmulatorRunner
from runners import Runners
import numpy as np

from schedules import LinearSchedule, PiecewiseSchedule
from replay_buffer import *
from paac import PAACLearner
from train import get_network_and_environment_creator
import logger_utils
import tables
from misc_utils import LazyFrames
from PIL import Image

def test_state_transition(o_t, o_tp1):
    img_o_t = Image.fromarray(o_t)
    img_o_tp1 = Image.fromarray(o_tp1)
    img_o_t.show()
    img_o_tp1.show()

class PDQFDLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(PDQFDLearner, self).__init__(network_creator, environment_creator, args)
        self.workers = args.emulator_workers
        self.stochastic = args.stochastic
        self.prioritized = args.prioritized
        self.exp_epsilon = LinearSchedule(args.max_global_steps,
                                   initial_p=args.exp_epsilon,
                                   final_p=0.0)
        self.demo_agent_folder = args.demo_agent_folder
        self.test_count = args.test_count
        self.noops = args.noops
        self.serial_episodes = args.serial_episodes
        self.demo_args = args
        self.demo_train_ratio = args.demo_train_ratio

        if not self.demo:
            self.demo_trans_size = 0

        # Create replay buffer
        approximate_num_iters = self.max_global_steps / 4
        if self.prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.demo_trans_size, self.prioritized_alpha)
            self.beta_schedule = LinearSchedule(approximate_num_iters, initial_p=self.prioritized_beta0, final_p=1.0)
            # Function to return transition priority bonus
            self.set_p_eps = lambda x: self.prioritized_eps_d if x < self.demo_trans_size else self.prioritized_eps
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.demo_trans_size)

        self.actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        self.y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        self.rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        self.states = None #np.zeros([self.max_local_steps] + list(self.state_shape), dtype=np.uint8)
        self.actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        self.episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

        self.summaries_op = tf.summary.merge_all()
        self.state_shape = None
        self.counter = 0

    @staticmethod
    def choose_next_actions(network, num_actions, states, session, eps, stochastic):
        network_output_q = session.run(network.output_layer_q, 
            feed_dict={network.input_ph: states})

        action_indices = PDQFDLearner.__sample_policy_action(network_output_q, num_actions, eps, stochastic)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions #, network_output_q

    @staticmethod
    def get_maxq_values(network, states, session):
        network_output_q = session.run(network.output_layer_q,
            feed_dict={network.input_ph: states})

        maxq_values = network_output_q.max(axis=-1)

        return maxq_values

    def __choose_next_actions(self, states):
        eps = self.exp_epsilon.value(self.global_step)
        return PDQFDLearner.choose_next_actions(self.network, self.num_actions, states, self.session, eps, self.stochastic)

    def __get_target_maxq_values(self, states):
        return PDQFDLearner.get_maxq_values(self.target_network, states, self.session)

    @staticmethod
    def __sample_policy_action(q_values, num_actions, eps, stochastic):
        """
        Sample an action using the Q values output by the Q network.
        """
        batch_size = q_values.shape[0]
        
        deterministic_actions = np.argmax(q_values, axis=1)
        random_actions = np.random.randint(low=0, high=num_actions, size=batch_size)
        choose_random = np.random.uniform(low=0.0, high=1.0, size=batch_size) < eps
        stochastic_actions = np.where(choose_random, random_actions, deterministic_actions)

        if stochastic:
            action_indices = stochastic_actions
        else:
            action_indices = deterministic_actions

        return action_indices

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def init_buffer_with_db(self):
        logging.debug("Adding demonstration data from db to replay buffer.")
        hdf5_file = tables.open_file(self.demo_db, mode='r')
        max_demo_size = hdf5_file.root.obs.shape[0]
        
        bsize = min(1000, max_demo_size)
        for i in range(0, min(max_demo_size, self.demo_trans_size), bsize):
            obs = hdf5_file.root.obs[i:(i + bsize + 1)]
            if i == 0:
                self.state_shape = obs[i].shape
            act_rews = hdf5_file.root.act_rew_done[i:(i + bsize + 1)]
            for j in range(obs.shape[0] - 1):
                self.replay_buffer.add(LazyFrames([obs[j].tolist()]), 
                                       act_rews[j][0], act_rews[j][1], 
                                       LazyFrames([obs[j + 1].tolist()]), 
                                       float(act_rews[j][2]))
            logging.debug("Added {} new samples.".format(obs.shape[0]))
        hdf5_file.close()

    def init_buffer_with_demo(self):
        logging.debug("Adding demonstration data from demo agent to replay buffer.")
        # Configure demo model/agent to load
        arg_file = os.path.join(self.demo_agent_folder, 'args.json')
        for k, v in logger_utils.load_args(arg_file).items():
            setattr(self.demo_args, k, v)
        self.demo_args.max_global_steps = 0
        df = self.demo_agent_folder
        self.demo_args.debugging_folder = '/tmp/logs'
        self.demo_args.device = self.device
        self.demo_args.random_start = False
        self.demo_args.single_life_episodes = False
        self.demo_args.actor_id = 0
        rng = np.random.RandomState(int(time.time()))
        self.demo_args.random_seed = rng.randint(1000)
        
        # Create demo agent and environment creation handlers
        demo_network_creator, demo_env_creator = get_network_and_environment_creator(self.demo_args)
        demo_network = demo_network_creator(name="local_learning")
        
        # Instantiate saver to restore demo agent params 
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local_learning'))
        
        # Create "single" instance of environment
        environment = demo_env_creator.create_environment(0)
    
        config = tf.ConfigProto()
        if 'gpu' in self.device:
            config.gpu_options.allow_growth = True
        
        # Create session, load params, run agent and fill buffer
        with tf.Session(config=config) as demo_sess:
            checkpoints_ = os.path.join(df, 'checkpoints')
            demo_network.init(checkpoints_, saver, demo_sess)
            
            state = environment.reset_with_noops(self.noops)
            self.state_shape = state.shape
            next_state = np.zeros_like(state)    
            for demo_data_counter in range(1, self.demo_trans_size + 1):
                action, _, _ = PAACLearner.choose_next_actions(demo_network, demo_env_creator.num_actions, np.expand_dims(state, axis=0), demo_sess)
                next_state, reward, done = environment.next(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                np.copyto(state, next_state)
                if demo_data_counter % 1000 == 0 and demo_data_counter != 0:
                    logging.debug("Added {} of {} demo transitions".format(str(demo_data_counter), str(self.demo_trans_size))) # + str() + " of " + str(args.demo_trans_size) + " demo transitions")
                if done:
                    state = environment.reset_with_noops(self.noops)

        logging.debug("Demonstration data ready to use in buffer.")
        # Simple testing to see if states are stored correctly
        # test_state_transition(self.replay_buffer._storage[-5][0], self.replay_buffer._storage[-5][3])
    
    def get_trajectories_from_buffer(self):
        if self.prioritized:
            experience = self.replay_buffer.sample_nstep(self.batch_size, beta=self.beta_schedule.value(self.global_step), n_step=self.max_local_steps)
            (__states_t, __actions, __rewards, __states_tp1, __dones, _weights, _batch_idxes, _mask_margin_loss) = experience
        else:
            experience = self.replay_buffer.sample_nstep(self.batch_size, n_step=self.max_local_steps)
            (__states_t, __actions, __rewards, __states_tp1, __dones, _batch_idxes, _mask_margin_loss) = experience
            _weights = np.ones((self.batch_size,))

        return (np.transpose(__states_t, (1, 0, 2, 3, 4)), 
                np.squeeze(np.transpose(__actions, (1, 0, 2, 3)), axis=2), 
                np.transpose(__rewards), 
                np.transpose(__states_tp1, (1, 0, 2, 3, 4)), 
                np.transpose(__dones), 
                _weights, 
                _batch_idxes, 
                _mask_margin_loss)

    def estimate_returns(self, next_state_maxq):
        estimated_return = np.copy(next_state_maxq)
        for t in reversed(range(self.max_local_steps)):
            estimated_return = self.rewards[t] + self.gamma * estimated_return * self.episodes_over_masks[t]
            self.y_batch[t] = np.copy(estimated_return)

    def run_train_step(self, mask_margin=0.0):
        flat_states = self.states.reshape([self.max_local_steps * self.emulator_counts] + list(self.states.shape)[2:])
        flat_y_batch = self.y_batch.reshape(-1)
        flat_actions = self.actions.reshape(self.max_local_steps * self.emulator_counts, self.num_actions)

        lr = self.get_lr()
        feed_dict = {self.network.input_ph: flat_states,
                     self.network.target_ph: flat_y_batch,
                     self.network.mask_margin_loss: mask_margin,
                     self.network.selected_action_ph: flat_actions,
                     self.learning_rate: lr}

        _, summaries = self.session.run(
            [self.train_step, self.summaries_op],
            feed_dict=feed_dict)

        self.summary_writer.add_summary(summaries, self.global_step)
        self.summary_writer.flush()

        self.counter += 1

    def train_with_buffer_experience(self, st, a, r, stp1, d, w, id, m=None, buff=None):
        # Set up stpn
        stpn = np.copy(stp1[-1, :])
        np.copyto(self.rewards, r)
        np.copyto(self.states, st)
        np.copyto(self.actions, a)
        np.copyto(self.episodes_over_masks, d)
        
        next_state_maxq = self.__get_target_maxq_values(stpn)        
        self.estimate_returns(next_state_maxq)
        self.run_train_step(mask_margin=1.0)


    def batched_demo_training(self):
        batched_steps = self.max_local_steps * self.emulator_counts
        (__s_t, __a, __r, __s_tp1, __dones, _w, _idx, _m) = self.get_trajectories_from_buffer()
        # print(__s_t.shape, __a.shape, __r.shape, __s_tp1.shape, __dones.shape, _w.shape, len(_idx), len(_m))
        # e.g. with Atari Breakout env, steps 5, batch size 32
        # (5, 32, 84, 84, 4) (5, 32, 4) (5, 32) (5, 32, 84, 84, 4) (5, 32) (32,) 32 32                  
        self.train_with_buffer_experience(__s_t, __a, __r, __s_tp1, __dones, _w, _idx, _m, buff=True)
        self.global_step += batched_steps

    
    def train(self):
        """
        Main actor learner loop for parallel deep Q learning with demonstrations.
        """

        # Fill replay buffer with demo data
        if self.demo_db is not None:
            self.init_buffer_with_db()
        else:
            self.init_buffer_with_demo()

        self.global_step = self.init_network()
        global_step_start = self.global_step

        self.update_target()
        
        self.states = np.zeros([self.max_local_steps, self.emulator_counts] + list(self.state_shape), dtype=np.uint8)

        # Pre-train using demonstration data
        if self.demo:
            logging.debug("Pre-training...")
            batched_steps = self.max_local_steps * self.emulator_counts
            for t in range(batched_steps, self.pre_train_steps, batched_steps):
                if t % 1000 == 0:
                    logging.debug("Pre-training step {} of {} complete".format(str(t), str(self.pre_train_steps)))
                
                self.batched_demo_training()
                
                if t % self.target_update_freq == 0:
                    self.update_target()

        # OPTIONS for proceeding to train with emulators
        # 1. Proceede collecting new experiences in buffer.
        #    Train using the buffer without overwriting demo. When using demo data,
        #    do not mask the supervised loss.
        # OR
        # 2. Train using parallel experiences from emulators. Mask the supervised loss.
        #    Alternate between using demo data from buffer and new emulator data 
        #    collected online in parallel.
        # Choosing OPTION 2.

        # Train using emulator data (TODO: and demo data)
        logging.debug("Resuming training from emulators at Step {}".format(self.global_step))
        total_rewards = []
        # state, reward, episode_over, action
        variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.uint8)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()
        shared_states, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        start_time = time.time()

        while self.global_step < self.max_global_steps - self.pre_train_steps:

            loop_start_time = time.time()

            if np.random.uniform(0, 1) < self.demo_train_ratio:
                # train with demo data
                logging.debug("Training using demo data at Step {}".format(self.global_step))
                self.batched_demo_training()
            else:
                # Collect experience
                for t in range(self.max_local_steps):
                    next_actions = self.__choose_next_actions(shared_states)
                    self.actions_sum += next_actions
                    for z in range(next_actions.shape[0]):
                        shared_actions[z] = next_actions[z]
    
                    self.actions[t] = next_actions
                    self.states[t] = shared_states
    
                    # Start updating all environments with next_actions
                    self.runners.update_environments()
                    self.runners.wait_updated()
                    # Done updating all environments, have new states, rewards and is_over
    
                    self.episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)
    
                    for e, (actual_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):
                        total_episode_rewards[e] += actual_reward
                        actual_reward = self.rescale_reward(actual_reward)
                        self.rewards[t, e] = actual_reward
    
                        emulator_steps[e] += 1
                        self.global_step += 1
                        if episode_over:
                            total_rewards.append(total_episode_rewards[e])
                            episode_summary = tf.Summary(value=[
                                tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[e]),
                                tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[e]),
                            ])
                            self.summary_writer.add_summary(episode_summary, self.global_step)
                            self.summary_writer.flush()
                            total_episode_rewards[e] = 0
                            emulator_steps[e] = 0
                            self.actions_sum[e] = np.zeros(self.num_actions)
    
                next_state_maxq = self.__get_target_maxq_values(shared_states)
                
                self.estimate_returns(next_state_maxq)
                self.run_train_step(mask_margin=0.0)

            if self.global_step % self.target_update_freq == 0:
                self.update_target()

            if self.counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_step_start) / (curr_time - start_time),
                                     last_ten))
            self.save_vars()

        self.cleanup()

    def cleanup(self):
        super(PDQFDLearner, self).cleanup()
        self.runners.stop()
