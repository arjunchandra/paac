import time
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *

from emulator_runner import EmulatorRunner
from runners import Runners
import numpy as np

from schedules import LinearSchedule, PiecewiseSchedule
from replay_buffer import *
from demo_agent_corridor import *
import logger_utils
import tables
from misc_utils import LazyFrames
from PIL import Image


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def test_state_transition(o_t, o_tp1):
    img_o_t = Image.fromarray(o_t)
    img_o_tp1 = Image.fromarray(o_tp1)
    img_o_t.show()
    img_o_tp1.show()

class SimplePDQFDLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(SimplePDQFDLearner, self).__init__(network_creator, environment_creator, args)

        self.debug = args.debug
        self.experiment_type = args.experiment_type

        self.double_q = args.double_q
        self.continuous_target_update = args.continuous_target_update
        self.stochastic = args.stochastic
        #self.exp_epsilon = LinearSchedule(args.max_global_steps,
        #                           initial_p=args.exp_epsilon,
        #                           final_p=0.0)
        #self.exp_epsilon = PiecewiseSchedule([(0, args.exp_epsilon), (round(args.max_global_steps/3), 0.3), (round(2*args.max_global_steps/3), 0.01)], outside_value=0.001)
        self.exp_epsilon = PiecewiseSchedule([(0, args.exp_epsilon), (10000, args.exp_epsilon), (30000, 0)],
                                             outside_value=0)
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
        self.n_trajectories = round(1.0 * args.batch_size / self.max_local_steps)
        self.replay_buffer_size = args.replay_buffer_size
        # Create replay buffer
        self.prioritized = args.prioritized
        if self.prioritized:
            self.prioritized_alpha = args.prioritized_alpha
            self.prioritized_beta0 = args.prioritized_beta0
            self.prioritized_eps = args.prioritized_eps
            self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, self.demo_trans_size, self.prioritized_alpha, n_emus=self.emulator_counts, debug =self.debug)
            self.beta_schedule = LinearSchedule(self.max_global_steps, initial_p=self.prioritized_beta0, final_p=1.0)
            # Function to return transition priority bonus
            self.set_p_eps = lambda x: self.prioritized_eps_d if x < self.demo_trans_size else self.prioritized_eps
        else:
            self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.demo_trans_size, n_emus=self.emulator_counts, debug =self.debug)

        self.y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        self.rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        self.states = np.zeros([self.max_local_steps, self.emulator_counts] + list(self.state_shape), dtype=np.float64)
        self.actions = np.zeros((self.max_local_steps, self.emulator_counts))
        self.episode_dones = np.zeros((self.max_local_steps, self.emulator_counts))

        tf.summary.scalar("Margin_loss", self.network.margin_loss)
        tf.summary.scalar("Regularization_loss", self.network.reg_loss)
        self.summaries_op = tf.summary.merge_all()
        self.state_shape = None
        self.counter = 0

        self.experiment_type = args.experiment_type



    @staticmethod
    def choose_next_actions(network, num_actions, states, session, eps, stochastic):
        network_output_q = session.run(network.output_layer_q, 
            feed_dict={network.input_ph: states})

        deterministic_actions = np.argmax(network_output_q, axis=1)
        if stochastic:
            batch_size = network_output_q.shape[0]
            random_actions = np.random.randint(low=0, high=num_actions, size=batch_size)
            choose_random = np.random.uniform(low=0.0, high=1.0, size=batch_size) < eps
            stochastic_actions = np.where(choose_random, random_actions, deterministic_actions)
            action_indices = stochastic_actions
        else:
            action_indices = deterministic_actions

        return action_indices

    def __choose_next_actions(self, states):
        eps = self.exp_epsilon.value(self.global_step)
        return SimplePDQFDLearner.choose_next_actions(self.network, self.num_actions, states, self.session, eps, self.stochastic)

    @staticmethod
    def get_target_maxq_values(target_network, states, session, double_q=True, learning_network=None, debug=False):

        if double_q:
            assert learning_network is not None, "Double Q-learning requires learning network for target calculation"

            [target_network_q, learning_network_q] = session.run([target_network.output_layer_q, learning_network.output_layer_q],
                                                  feed_dict={target_network.input_ph: states, learning_network.input_ph: states})

            idx_best_action_from_learning_network = np.argmax(learning_network_q, axis=1)
            maxq_values = target_network_q[range(target_network_q.shape[0]), idx_best_action_from_learning_network]

            if debug:
                print("learning_network_q ", learning_network_q)
                print("idx_best_action_from_learning_network ", idx_best_action_from_learning_network)
                print("target_network_q", target_network_q)
                print("maxq_values ", maxq_values)
        else:
            target_network_q = session.run(target_network.output_layer_q,
                                           feed_dict={target_network.input_ph: states})
            maxq_values = target_network_q.max(axis=-1)

        return maxq_values

    def __get_target_maxq_values(self, states):
        return SimplePDQFDLearner.get_target_maxq_values(self.target_network, states, self.session, double_q=self.double_q, learning_network=self.network, debug=self.debug)


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

    def init_buffer_with_db(self):
        logger.debug("Adding demonstration data from db to replay buffer.")
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
            logger.debug("Added {} new samples.".format(obs.shape[0]))
        hdf5_file.close()


    def init_buffer_with_demo_simple(self):
        logger.debug("Adding demonstration data from demo agent to replay buffer.")
        run_demo_agent(self.demo_agent_folder, self.replay_buffer, self.demo_trans_size)


    def init_buffer_with_demo(self):
        logger.debug("Adding demonstration data from demo agent to replay buffer.")
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
                actual_reward = self.rescale_reward(reward)
                self.replay_buffer.add(state, action, actual_reward, next_state, done)
                np.copyto(state, next_state)
                if ((demo_data_counter % 1000 == 0 and demo_data_counter != 0) 
                    or (demo_data_counter == self.demo_trans_size)):
                    logger.debug("Added {} of {} demo transitions".format(str(demo_data_counter), str(self.demo_trans_size))) # + str() + " of " + str(args.demo_trans_size) + " demo transitions")
                if done:
                    state = environment.reset_with_noops(self.noops)

        logger.debug("Demonstration data ready to use in buffer.")
        # Simple testing to see if states are stored correctly
        # test_state_transition(self.replay_buffer._storage[-5][0], self.replay_buffer._storage[-5][3])


    def get_trajectories_from_buffer(self):
        if self.prioritized:
            experience = self.replay_buffer.sample_nstep(self.n_trajectories, self.max_local_steps, self.beta_schedule.value(self.global_step))
            (__states_t, __actions, __rewards, __states_tp1, __dones, _weights, _batch_idxes, _mask_margin_loss) = experience
        else:
            experience = self.replay_buffer.sample_nstep(self.n_trajectories, self.max_local_steps)
            (__states_t, __actions, __rewards, __states_tp1, __dones, _batch_idxes, _mask_margin_loss) = experience
            _weights = np.ones((self.emulator_counts,))

        return (__states_t, __actions, __rewards, __states_tp1, __dones, _weights, _batch_idxes, _mask_margin_loss)


    def estimate_returns(self, next_state_maxq, rewards, episode_dones):
        estimated_return = next_state_maxq
        episodes_over_masks = 1.0 - episode_dones.astype(np.float32)
        y = np.zeros_like(rewards)
        if self.use_exp_replay:
            for t in reversed(range(self.max_local_steps)):
                estimated_return = rewards[:, t] + self.gamma * estimated_return * episodes_over_masks[:, t]
                y[:, t] = estimated_return
        else:
            for t in reversed(range(self.max_local_steps)):
                estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
                y[t] = estimated_return
        return y

    def run_train_step(self, states, actions, targets, imp_weights, mask_margin=0.0):
        n = self.n_trajectories if self.use_exp_replay else self.emulator_counts
        states = np.reshape(states, [self.max_local_steps * n] + list(states.shape)[2:])
        actions = np.reshape(actions, -1)
        targets = np.reshape(targets, -1)

        lr = self.get_lr()
        feed_dict = {self.network.input_ph: states,
                     self.network.target_ph: targets,
                     self.network.importance_weights_ph: imp_weights,
                     self.network.mask_margin_loss: mask_margin,
                     self.network.selected_action_ph: actions,
                     self.learning_rate: lr}

        _, output_layer_q, output_selected_action, loss, td_error, wt_margin_loss, summaries = self.session.run(
            [self.train_step, self.network.output_layer_q, self.network.output_selected_action, self.network.loss,
             self.network.td_error, self.network.wt_margin_loss, self.summaries_op],
            feed_dict=feed_dict)

        if self.debug:
            print("output_layer_q ", output_layer_q.shape, output_layer_q)
            print("output_selected_action ", output_selected_action.shape, output_selected_action)
            print("td_error ", td_error.shape, td_error)
            print("wt_margin_loss ", wt_margin_loss.shape, wt_margin_loss)
            print("loss ", loss)

        self.summary_writer.add_summary(summaries, self.global_step)
        self.summary_writer.flush()

        self.counter += 1

        return td_error


    def train_from_experience(self, shared_states, mask_margin=0.0):
        if self.use_exp_replay:
            # demo_data_masks is 1 for each sample/trajectory coming from demo data, and 0 otherwise
            (s_t, a, r, s_tp1, dones, imp_weights, idxes, demo_data_masks) = self.get_trajectories_from_buffer()

            if self.debug:
                for i in range(s_t.shape[0]):
                    x = []
                    for j in range(s_t.shape[1]):
                        x.append(np.argmax(s_t[i, j, :].flatten()))
                        pass
                    print("FROM BUFFER: St: ", x)
                    pass
                print("FROM BUFFER: Actions ", a)
                print("DEMO MASKS: ", demo_data_masks)
                print("IDXES: ", idxes)

            # Calculate returns for all trajectories
            s_tpn = s_tp1[:, -1, :]
            next_state_maxq = self.__get_target_maxq_values(s_tpn)
            targets = self.estimate_returns(next_state_maxq, r, dones)

            if self.debug:
                print("Epi masks ", 1.0 - dones.astype(np.float32))
                print("Rewards ", r)
                print("Targets ", targets)

            td_errors = self.run_train_step(s_t, a, targets, imp_weights, mask_margin=demo_data_masks)

            if self.prioritized:
                # Add bonus to demo transition priorities
                # Obs! The indexes refer to the first sample on each trajectory. We can use the index of the first sample
                # for all samples in a trajectory, in order to differentiate between demo and self-generated data
                p_eps = np.array([self.set_p_eps(idx) for idx in idxes]) #for _ in range(self.max_local_steps) ])
                if self.debug:
                    print("P_EPS: ", p_eps)
                    print("Shapes td_errors, p_eps: ", td_errors.shape, p_eps.shape)
                new_priorities = np.abs(td_errors) + p_eps
                self.replay_buffer.update_priorities(idxes, new_priorities)
        else:
            next_state_maxq = self.__get_target_maxq_values(shared_states)
            targets = self.estimate_returns(next_state_maxq, self.rewards, self.episode_dones)
            imp_weights = np.ones_like(targets)
            self.run_train_step(self.states, self.actions, targets, imp_weights, mask_margin=mask_margin)

    def collect_experience(self, shared_states, shared_actions, shared_rewards, shared_episode_over):
        for t in range(self.max_local_steps):
            next_actions = self.__choose_next_actions(shared_states)
            #self.actions_sum += next_actions
            for z in range(next_actions.shape[0]):
                shared_actions[z] = next_actions[z]

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
                self.global_step += 1
                if episode_over:
                    self.emu_n_epi[emu] += 1
                    self.emu_acc_reward[emu] += self.total_episode_rewards[emu]
                    self.emu_epi_succ[emu] += self.total_episode_rewards[emu]
                    if self.emu_n_epi[emu] % 100 == 0:
                        logger.debug("{} steps. Emu {}: Epi. succ. rate: {:.2f}%, Epi. reward: {}, Acc. reward: {}, Epsilon: {:.3f}".format(self.global_step,
                                                                                                                                            emu,
                                                                                                                                            self.emu_epi_succ[
                                                                                                                                                emu]/self.goal_reward,
                                                                                                                                            self.total_episode_rewards[
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


        if self.use_exp_replay:
            for emu in range(self.emulator_counts):
                if self.debug:
                    s = []
                    for t in range((self.max_local_steps)):
                         s.append(np.argmax(self.states[t,emu].flatten()))
                         pass
                    print("TO BUFFER emu {}: States: {}".format(emu, s))
                    print("TO BUFFER emu {}: Actions: {}".format(emu, self.actions[:, emu]))

                for t in range((self.max_local_steps -1)):
                    self.replay_buffer.add(np.copy(self.states[t, emu]), np.copy(self.actions[t, emu]), np.copy(self.rewards[t, emu]),
                                           np.copy(self.states[t + 1, emu]), np.copy(self.episode_dones[t, emu]), emu)

                t = self.max_local_steps - 1 # Obs! Subtract 1 due to zero-based indexing
                self.replay_buffer.add(np.copy(self.states[t, emu]), np.copy(self.actions[t, emu]), np.copy(self.rewards[t, emu]),
                                       np.copy(shared_states[emu]), np.copy(self.episode_dones[t, emu]), emu)

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

        # Average reward over 100 episodes
        self.goal_reward = 1 if self.experiment_type == 'corridor' else 500

        if self.demo:
            if self.experiment_type == 'cartpole':
                import gym
                eva_env = gym.make("CartPole-v1")
                _succ_epi = evaluate(eva_env, self.session, self.network.output_layer_q, self.network.input_ph,
                                               visualize=False, v_func=self.network.value, goal_reward=self.goal_reward)
                logger.debug(
                    "Before pre-training - Evaluation success rate over 100 episodes: {:.2f}%".format(
                        _succ_epi))

                import pickle
                with open('demo.p', 'rb') as f:
                    buffer = pickle.load(f)
                for sample in buffer:
                    self.replay_buffer.add_demo(sample[0], sample[1], sample[2], sample[3], sample[4])

            elif self.experiment_type == 'corridor':
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
                eva_env = GymEnvironment(-1, envs[env_id][0], 3, env_class=envs[env_id][1], visualize=False,
                                         agent_history_length=1)
                _succ_epi = evaluate(eva_env, self.session, self.network.output_layer_q, self.network.input_ph,
                                               visualize=False, v_func=self.network.value, goal_reward=self.goal_reward)
                logger.debug(
                    "Before pre-training - Evaluation success rate over 100 episodes: {:.2f}%".format(_succ_epi))

                self.init_buffer_with_demo_simple()


            # Pre-train using demonstration data
            if self.global_step < self.pre_train_steps:
                logger.debug("Pre-training...")
                resume_step = self.global_step // (self.n_trajectories * self.max_local_steps)
                for t in range(resume_step, self.pre_train_steps):
                    if t % 1000 == 0:
                        logger.debug(
                            "Pre-training batch step {} of {} complete".format(str(t), str(self.pre_train_steps)))

                    self.train_from_experience(None, mask_margin=1.0)

                    if self.continuous_target_update or t % self.target_update_freq == 0:
                        self.update_target()

                    self.global_step += self.n_trajectories * self.max_local_steps

                _succ_epi = evaluate(eva_env, self.session, self.network.output_layer_q, self.network.input_ph, visualize=False, v_func=self.network.value, goal_reward=self.goal_reward)
                logger.debug("After pre-training - Evaluation success rate over 100 episodes: {:.2f}%".format(_succ_epi))


        logger.debug("Resuming training from emulators at Step {}".format(self.global_step))
        self.total_rewards = []
        self.emulator_steps = [0] * self.emulator_counts
        self.total_episode_rewards = self.emulator_counts * [0]
        self.emu_acc_reward = [0 for _ in range(self.emulator_counts)]
        self.emu_epi_succ = [0 for _ in range(self.emulator_counts)]
        self.emu_n_epi = [0 for _ in range(self.emulator_counts)]


        # state, reward, episode_over, action
        variables = [(np.asarray([emulator.reset() for emulator in self.emulators], dtype=np.float64)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros(self.emulator_counts, dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()
        shared_states, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        start_time = time.time()
        while self.global_step < self.max_global_steps - self.pre_train_steps:

            loop_start_time = time.time()

            # Collect experience
            self.collect_experience(shared_states, shared_actions, shared_rewards, shared_episode_over)
    
            if self.global_step > self.learning_start:
                self.train_from_experience(shared_states)
                #print(self.global_step)
                #sys.exit(0)

            if (self.global_step > self.learning_start) and (
                    self.continuous_target_update or self.global_step % self.target_update_freq == 0):
                self.update_target()

            self.save_vars()

        _succ_epi, _acc_rew = evaluate(eva_env, self.session, self.network.output_layer_q, self.network.input_ph,
                                       visualize=True, v_func=self.network.value)
        logger.debug(
            "End - Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi, _acc_rew))


        self.cleanup()

    def cleanup(self):
        super(SimplePDQFDLearner, self).cleanup()
        self.runners.stop()
