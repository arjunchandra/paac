import time
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *

from emulator_runner import EmulatorRunner
from runners import Runners
import numpy as np
import random

from schedules import LinearSchedule, PiecewiseSchedule
from replay_buffer import *
# from paac import PAACLearner
# from train import get_network_and_environment_creator
from demo_agent_corridor import *
import logger_utils
import tables
from misc_utils import LazyFrames
from PIL import Image


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class my_experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def test_state_transition(o_t, o_tp1):
    img_o_t = Image.fromarray(o_t)
    img_o_tp1 = Image.fromarray(o_tp1)
    img_o_t.show()
    img_o_tp1.show()



class SimplePDQFDLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(SimplePDQFDLearner, self).__init__(network_creator, environment_creator, args)


        self.double_q = args.double_q
        self.continuous_target_update = args.continuous_target_update
        self.stochastic = args.stochastic
        #self.exp_epsilon = LinearSchedule(args.max_global_steps,
        #                           initial_p=args.exp_epsilon,
        #                           final_p=0.0)
        self.exp_epsilon = PiecewiseSchedule([(0, args.exp_epsilon), (10000, args.exp_epsilon), (50000, 0)], outside_value=0)

        self.demo = args.demo
        self.demo_db = args.demo_db
        self.demo_trans_size = args.demo_trans_size
        self.demo_model_dir = args.demo_model_dir
        self.pre_train_steps = args.pre_train_steps
        self.prioritized_eps_d = args.prioritized_eps_d

        self.demo_agent_folder = args.demo_agent_folder
        self.test_count = args.test_count
        self.noops = args.noops
        self.serial_episodes = args.serial_episodes
        self.demo_args = args
        self.demo_train_ratio = args.demo_train_ratio

        self.learning_start = self.pre_train_steps
        if not self.demo:
            self.demo_trans_size = 0
            self.pre_train_steps = 0
            self.demo_train_ratio = 0
            self.learning_start = 10000

        # Replay buffer
        self.use_exp_replay = args.use_exp_replay
        self.replay_buffer_size = args.replay_buffer_size
        # Create replay buffer
        self.prioritized = args.prioritized
        if self.prioritized:
            self.prioritized_alpha = args.prioritized_alpha
            self.prioritized_beta0 = args.prioritized_beta0
            self.prioritized_eps = args.prioritized_eps
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.demo_trans_size, self.prioritized_alpha)
            self.beta_schedule = LinearSchedule(self.max_global_steps, initial_p=self.prioritized_beta0, final_p=1.0)
            # Function to return transition priority bonus
            self.set_p_eps = lambda x: self.prioritized_eps_d if x < self.demo_trans_size else self.prioritized_eps
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.demo_trans_size)

        self.actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        self.y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        self.rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        self.states = np.zeros([self.max_local_steps, self.emulator_counts] + list(self.state_shape))
        self.actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        self.episode_dones = np.zeros((self.max_local_steps, self.emulator_counts))

        self.summaries_op = tf.summary.merge_all()
        self.state_shape = None
        self.counter = 0

        self.experiment_type = args.experiment_type

    @staticmethod
    def __sample_policy_action(q_values, num_actions, eps, stochastic):
        """
        Sample an action using the Q values output by the Q network.
        """
        batch_size = q_values.shape[0]

        deterministic_actions = np.argmax(q_values, axis=1)
        if stochastic:
            random_actions = np.random.randint(low=0, high=num_actions, size=batch_size)
            choose_random = np.random.uniform(low=0.0, high=1.0, size=batch_size) < eps
            stochastic_actions = np.where(choose_random, random_actions, deterministic_actions)
            action_indices = stochastic_actions
        else:
            action_indices = deterministic_actions

        return action_indices

    @staticmethod
    def choose_next_actions(network, num_actions, states, session, eps, stochastic):
        network_output_q = session.run(network.output_layer_q, 
            feed_dict={network.input_ph: states})

        action_indices = SimplePDQFDLearner.__sample_policy_action(network_output_q, num_actions, eps, stochastic)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions #, network_output_q

    def __choose_next_actions(self, states):
        eps = self.exp_epsilon.value(self.global_step)
        return SimplePDQFDLearner.choose_next_actions(self.network, self.num_actions, states, self.session, eps, self.stochastic)

    @staticmethod
    def get_target_maxq_values(target_network, states, session, double_q=True, learning_network=None):

        if double_q:
            assert learning_network is not None, "Double Q-learning requires learning network for target calculation"

            [target_network_q, learning_network_q] = session.run([target_network.output_layer_q, learning_network.output_layer_q],
                                                  feed_dict={target_network.input_ph: states, learning_network.input_ph: states})

            ##print("learning_network_q ", learning_network_q)
            idx_best_action_from_learning_network = np.argmax(learning_network_q, axis=1)
            ##print("idx_best_action_from_learning_network ", idx_best_action_from_learning_network)
            ##print("target_network_q", target_network_q)
            maxq_values = target_network_q[range(target_network_q.shape[0]), idx_best_action_from_learning_network]
            ##print("maxq_values ", maxq_values)
        else:
            target_network_q = session.run(target_network.output_layer_q,
                                           feed_dict={target_network.input_ph: states})
            maxq_values = target_network_q.max(axis=-1)

        return maxq_values

    def __get_target_maxq_values(self, states):
        return SimplePDQFDLearner.get_target_maxq_values(self.target_network, states, self.session, double_q=self.double_q, learning_network=self.network)


    def update_target(self):
        if self.continuous_target_update:
            self.session.run(self.target_network.continuous_sync_nets)
        else:
            params = self.network.get_params(self.session)
            feed_dict = {}
            for i in range(len(self.target_network.params)):
                feed_dict[self.target_network.params_ph[i]] = params[i]
            self.target_network.set_params(feed_dict, self.session)


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


    def get_trajectories_from_buffer(self):
        if self.prioritized:
            experience = self.replay_buffer.sample_nstep(self.emulator_counts, beta=self.beta_schedule.value(self.global_step), n_step=self.max_local_steps)
            (__states_t, __actions, __rewards, __states_tp1, __dones, _weights, _batch_idxes, _mask_margin_loss) = experience
        else:
            experience = self.replay_buffer.sample_nstep(self.emulator_counts, n_step=self.max_local_steps)
            (__states_t, __actions, __rewards, __states_tp1, __dones, _batch_idxes, _mask_margin_loss) = experience
            _weights = np.ones((self.emulator_counts,))


        # if self.experiment_type == 'atari':
        #     return (np.transpose(__states_t, (1, 0, 2, 3, 4)),
        #             np.squeeze(np.transpose(__actions, (1, 0, 2, 3)), axis=2),
        #             np.transpose(__rewards),
        #             np.transpose(__states_tp1, (1, 0, 2, 3, 4)),
        #             np.transpose(__dones),
        #             _weights,
        #             _batch_idxes,
        #             _mask_margin_loss)
        # else: # corridor
        #     return (np.transpose(__states_t, (1, 0, 2)),
        #             np.transpose(__actions, (1, 0, 2)),
        #             np.transpose(__rewards),
        #             np.transpose(__states_tp1, (1, 0, 2)),
        #             np.transpose(__dones),
        #             _weights,
        #             _batch_idxes,
        #             _mask_margin_loss)

        return (np.swapaxes(__states_t, 0, 1),
                    np.swapaxes(__actions, 0, 1),
                    np.transpose(__rewards),
                    np.swapaxes(__states_tp1, 0, 1),
                    np.transpose(__dones),
                    _weights,
                    _batch_idxes,
                    _mask_margin_loss)




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

    def train_from_replay_buffer(self, buffer, mask_margin=0.0):
        trainBatch = buffer.sample(32)

#        for i in range(3):
#            #print("Buffer read: ", np.argmax(trainBatch[i,0].flatten()), np.argmax(trainBatch[i,1]),
#                  trainBatch[i, 2],
#              np.argmax(trainBatch[i,3].flatten()), trainBatch[i,4])

        states = np.stack(trainBatch[:, 0])
        actions= np.stack(trainBatch[:, 1])
        rewards = np.stack(trainBatch[:,2])
        episode_dones = np.stack(trainBatch[:, 4])

        ##print(trainBatch[:, 3].shape, trainBatch[0, 3].shape)

        s_tpn = np.stack(trainBatch[:, 3])
        #print("s_tpn", s_tpn.shape)
        #print("s_t", states.shape)
        next_state_maxq = self.__get_target_maxq_values(s_tpn)

        #print("next_state_maxq ", next_state_maxq.shape, next_state_maxq)
        episodes_over_masks = 1.0 - episode_dones.astype(np.float32)
        #print("episodes_over_masks ", episodes_over_masks.shape, episodes_over_masks)
        #print("rewards: ", rewards, rewards.shape)
        y_batch = rewards + self.gamma * next_state_maxq * episodes_over_masks
        #print("y_batch ", y_batch)

        flat_states = states #.reshape([self.max_local_steps * self.emulator_counts] + list(self.states.shape)[2:])
        flat_y_batch = y_batch.reshape(-1)
        #print("flat_y_batch ", flat_y_batch)
        #print("actions", actions, actions.shape)
        flat_actions = actions#.reshape((3, self.num_actions))
        #print("flat_actions ",flat_actions, flat_actions.shape)

        lr = self.get_lr()
        feed_dict = {self.network.input_ph: flat_states,
                     self.network.target_ph: flat_y_batch,
                     self.network.mask_margin_loss: mask_margin,
                     self.network.selected_action_ph: flat_actions,
                     self.learning_rate: lr}

        _, output_layer_q, output_selected_action, td_error, loss, summaries = self.session.run(
            [self.train_step, self.network.output_layer_q, self.network.output_selected_action, self.network.td_error, self.network.loss, self.summaries_op],
            feed_dict=feed_dict)

        #print("output_layer_q ", output_layer_q.shape, output_layer_q)
        #print("output_selected_action ", output_selected_action.shape, output_selected_action)
        #print("td_error ", td_error.shape, td_error)
        #print("loss ", loss)

        self.summary_writer.add_summary(summaries, self.global_step)
        self.summary_writer.flush()

        self.counter += 1



    # def train_with_buffer_experience(self, st, a, r, stp1, d, w, id, m=None, mask_margin=0.0):
    #     # Set up stpn
    #     stpn = np.copy(stp1[-1, :])
    #     np.copyto(self.rewards, r)
    #     np.copyto(self.states, st)
    #     np.copyto(self.actions, a)
    #     np.copyto(self.episodes_over_masks, d)
    #
    #     next_state_maxq = self.__get_target_maxq_values(stpn)
    #     self.estimate_returns(next_state_maxq)
    #     self.run_train_step(mask_margin=mask_margin)

    # def batched_demo_training(self):
    #     batched_steps = self.max_local_steps * self.emulator_counts
    #     (__s_t, __a, __r, __s_tp1, __dones, _w, _idx, _m) = self.get_trajectories_from_buffer()
    #     # #print(__s_t.shape, __a.shape, __r.shape, __s_tp1.shape, __dones.shape, _w.shape, len(_idx), len(_m))
    #     # e.g. with Atari Breakout env, steps 5, batch size 32
    #     # (5, 32, 84, 84, 4) (5, 32, 4) (5, 32) (5, 32, 84, 84, 4) (5, 32) (32,) 32 32
    #     self.train_with_buffer_experience(__s_t, __a, __r, __s_tp1, __dones, _w, _idx, _m, buff=True)
    #     self.global_step += batched_steps

    def train_from_experience(self, shared_states, buffer):
        if self.use_exp_replay:
            self.train_from_replay_buffer(buffer)
        else:
            next_state_maxq = self.__get_target_maxq_values(shared_states)
            self.estimate_returns(next_state_maxq)
            self.run_train_step(mask_margin=0.0)

    def collect_experience(self, shared_states, shared_actions, shared_rewards, shared_episode_over, buffer):
        for t in range(self.max_local_steps):
            ##print("Current state: ", np.argmax(shared_states[0].flatten()))
            next_actions = self.__choose_next_actions(shared_states)
            self.actions_sum += next_actions
            for z in range(next_actions.shape[0]):
                shared_actions[z] = next_actions[z]
            ##print("Next/Shared actions: ", np.argmax(shared_actions[0]))

            self.actions[t] = next_actions
            self.states[t] = shared_states

            # Start updating all environments with next_actions
            self.runners.update_environments()
            self.runners.wait_updated()
            # Done updating all environments, have new states, rewards and is_over in shared_X variables

            self.episode_dones[t] = shared_episode_over

            for emu, (actual_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):
                self.total_episode_rewards[emu] += actual_reward
                self.rewards[t, emu] = self.rescale_reward(actual_reward)

                self.emulator_steps[emu] += 1
                if episode_over:
                    self.emu_n_epi[emu] += 1
                    self.emu_acc_reward[emu] += self.total_episode_rewards[emu]
                    if actual_reward == self.goal_reward: self.emu_epi_succ[emu] += 1
                    if self.emu_n_epi[emu] % 100 == 0:
                        logger.debug("Steps {}: Epi. succ. rate: {}%, Acc. reward: {}, Epsilon: {}".format(self.global_step,
                                                                                                         self.emu_epi_succ[
                                                                                                             emu],
                                                                                                         self.emu_acc_reward[
                                                                                                             emu],
                                                                                                         self.exp_epsilon.value(
                                                                                                             self.global_step)))

                        # emu_n_epi[emu] = 0
                        self.emu_epi_succ[emu] = 0

                    self.total_rewards.append(self.total_episode_rewards[emu])
                    episode_summary = tf.Summary(value=[
                        tf.Summary.Value(tag='rl/reward', simple_value=self.total_episode_rewards[emu]),
                        tf.Summary.Value(tag='rl/episode_length', simple_value=self.emulator_steps[emu]),
                    ])
                    self.summary_writer.add_summary(episode_summary, self.global_step)
                    self.summary_writer.flush()
                    self.total_episode_rewards[emu] = 0
                    self.emulator_steps[emu] = 0
                    self.actions_sum[emu] = np.zeros(self.num_actions)

            if self.use_exp_replay:
                for emu in range(self.emulator_counts):
                    t = 0
                    buffer.add(np.reshape(np.array([np.copy(self.states[t, emu]), np.copy(self.actions[t, emu]), np.copy(self.rewards[t, emu]),
                                           np.copy(shared_states[emu]), np.copy(self.episode_dones[t, emu])]), [1, 5]))
                    ##print("Buffer add: ", np.argmax(self.states[t, emu].flatten()), np.argmax(self.actions[t, emu]), self.rewards[t, emu],
                    #                       np.argmax(shared_states[emu].flatten()), self.episode_dones[t, emu] )

    def train(self):
        """
        Main actor learner loop for parallel deep Q learning with demonstrations.
        """
        batched_steps = self.max_local_steps * self.emulator_counts
        # Initialize networks
        self.global_step = self.init_network()
        global_step_start = self.global_step

        self.update_target()
        logging.info("Synchronized learning and target networks")

        envs = {
            1: ('FrozenLake-v0', None, 1), # Last element reflects the reward received if the end goal is reached
            2: ('FrozenLakeNonskid4x4-v0', None, 1),
            3: ('FrozenLakeNonskid8x8-v0', None, 1),
            4: ('CorridorSmall-v1', CorridorEnv, 1),
            5: ('CorridorSmall-v2', CorridorEnv, 1),
            6: ('CorridorActionTest-v0', CorridorEnv, 1),
            7: ('CorridorActionTest-v1', ComplexActionSetCorridorEnv, 1),
            8: ('CorridorBig-v0', CorridorEnv, 1),
            9: ('CorridorFLNonSkid-v1', CorridorEnv, 1)
        }
        env_id = 9
        self.goal_reward = envs[env_id][2]
        eva_env = GymEnvironment(-1, envs[env_id][0], env_class=envs[env_id][1], visualize=False,
                                 agent_history_length=1)
        _succ_epi, _acc_rew = evaluate(eva_env, self.session, self.network.output_layer_q, self.network.input_ph,
                                       visualize=False, v_func=self.network.value)
        logger.debug(
            "Start - Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi, _acc_rew))

        myBuffer = my_experience_buffer()

        logger.debug("Resuming training from emulators at Step {}".format(self.global_step))
        self.total_rewards = []
        self.emulator_steps = [0] * self.emulator_counts
        self.total_episode_rewards = self.emulator_counts * [0]
        self.emu_acc_reward = [0 for _ in range(self.emulator_counts)]
        self.emu_epi_succ = [0 for _ in range(self.emulator_counts)]
        self.emu_n_epi = [0 for _ in range(self.emulator_counts)]

        # state, reward, episode_over, action
        variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.float64)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()
        shared_states, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        start_time = time.time()
        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()


            # Collect experience
            self.collect_experience(shared_states, shared_actions, shared_rewards, shared_episode_over, myBuffer)

            if (self.global_step > self.learning_start) and (self.global_step % 4 == 0):
                ##print("TRAIN")
                self.train_from_experience(shared_states, myBuffer)
                self.update_target()


            #self.save_vars()
            self.global_step += 1

        _succ_epi, _acc_rew = evaluate(eva_env, self.session, self.network.output_layer_q, self.network.input_ph,
                                       visualize=False, v_func=self.network.value)
        logger.debug(
            "End - Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi, _acc_rew))


        self.cleanup()

    def cleanup(self):
        super(SimplePDQFDLearner, self).cleanup()
        self.runners.stop()
