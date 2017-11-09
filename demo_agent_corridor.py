from __future__ import division

import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import os

from collections import defaultdict
import matplotlib

matplotlib.use("Qt5Agg")
import pylab
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
from pylab import rcParams
from corridor_emulator import *


# Plot out the values the critic gives for the agent being in
# a specific state, i.e. in a specific location in the env.
def plot_value_function(sess, q_max, v_func, input_ph, env):
    grid_shape = env.grid_shape
    rcParams['figure.figsize'] = 1.2 * grid_shape[0], 1.2 * grid_shape[1]
    history_length = env.agent_history_length
    V = defaultdict(float)
    grid = np.zeros(grid_shape)
    for x in range(0, grid_shape[0]):
        for y in range(0, grid_shape[1]):
            if env.desc[x, y] == b'H':
                grid[x, y] = 0
            else:
                s = np.zeros(grid_shape)
                # Place the player at a given X/Y location.
                s[x, y] = 1
                s = s.flatten()
                s = np.reshape(s, (s.shape[0], 1))
                _s = np.concatenate([s for _ in range(history_length)], axis=1)
                grid[x, y] = sess.run(v_func, feed_dict={input_ph: [_s]})[0]
            V[(x, y)] = grid[x, y]

    def plot_value_function_3d(V, title="Value Function"):
        """
        Plots the value function as a surface plot.
        """
        min_x = min(k[0] for k in V.keys())
        max_x = max(k[0] for k in V.keys())
        min_y = min(k[1] for k in V.keys())
        max_y = max(k[1] for k in V.keys())

        x_range = np.arange(min_x, max_x + 1)
        y_range = np.arange(min_y, max_y + 1)
        X, Y = np.meshgrid(x_range, y_range)

        # Find value for all (x, y) coordinates
        Z = np.apply_along_axis(lambda _: V[(_[0], _[1])], 2, np.dstack([X, Y]))

        def plot_surface(X, Y, Z, title):
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                   cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
            ax.set_zlabel('Value')
            ax.set_title(title)
            ax.view_init(ax.elev, -120)
            fig.colorbar(surf)
            plt.show()

        plot_surface(X, Y, Z, "{}".format(title))

    def plot_optimal_actions(sess, q_max, input_ph, grid_shape, history_length=1):
        if env.game in ['CorridorActionTest-v1']:
            actU = [-1, 0, 0, 0, 1, 0, 0, 0]
            actV = [0, -1, -1, -1, 0, 1, 1, 1]
        else:
            actU = [-1, 0, 1, 0]
            actV = [0, -1, 0, 1]

        X = []
        Y = []
        U = []
        V = []
        for x in range(0, grid_shape[0]):
            for y in range(0, grid_shape[1]):
                if env.desc[x, y] != b'H':
                    s = np.zeros(grid_shape)
                    # Place the player at a given X/Y location.
                    s[x, y] = 1
                    s = s.flatten()
                    s = np.reshape(s, (s.shape[0], 1))
                    _s = np.concatenate([s for _ in range(history_length)], axis=1)  # .flatten()
                    action = sess.run(q_max, feed_dict={input_ph: [_s]})[0]
                    action = np.argmax(action)
                    # X += [y+0.5]
                    # Y += [x+0.5]
                    X += [y + 0.4]
                    Y += [grid_shape[0] - 0.5 - x]
                    U += [actU[action]]
                    V += [actV[action]]
                    try:
                        i = int(env.action_name(action)[0])
                        plt.text(y + 0.6, grid_shape[0] - 0.5 - x, i, fontweight='bold', horizontalalignment='center',
                                 verticalalignment='top')
                    except ValueError:
                        pass

        plt.quiver(X, Y, U, V)
        plt.show()

    # plot_value_function_3d(V, title="Value Function")

    # pylab.pcolor(grid)
    pylab.pcolormesh(np.flipud(grid))  # , cmap='RdBu')
    pylab.title("Value Function")
    pylab.colorbar()
    # pylab.xlabel("X")
    # pylab.ylabel("Y")
    # pylab.gca().invert_yaxis()
    pylab.draw()

    for x in range(0, grid_shape[0]):
        for y in range(0, grid_shape[1]):
            if env.desc[x, y] != b'H':
                plt.text(y + 0.5, grid_shape[0] - 0.8 - x, env.desc[x, y].decode('utf-8'), fontweight='bold',
                         horizontalalignment='center', verticalalignment='top')

    plot_optimal_actions(sess, q_max, input_ph, grid_shape, history_length=history_length)


class Qnetwork():
    def __init__(self, input_shape, hiddens, num_actions, dueling_type='avg', layer_norm=False):
        with tf.device('/cpu:0'):
            # with tf.name_scope('corridor_demo_agent'):
            if True:
                self.input = tf.placeholder(shape=(None, np.prod(input_shape)), dtype=tf.float32)
                out = self.input
                for hidden in hiddens:
                    out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        out = layers.layer_norm(out, center=True, scale=True)
                    out = tf.nn.relu(out)

                self.Advantage = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
                self.Value = layers.fully_connected(out, num_outputs=1, activation_fn=None)

                if dueling_type == 'avg':
                    self.Qout = self.Value + tf.subtract(self.Advantage,
                                                         tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
                elif dueling_type == 'max':
                    self.Qout = self.Value + tf.subtract(self.Advantage,
                                                         tf.reduce_max(self.Advantage, axis=1, keep_dims=True))
                else:
                    assert False, "dueling_type must be one of {'avg','max'}"

                self.predict = tf.argmax(self.Qout, 1)

                # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
                self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

                self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

                self.td_error = tf.square(self.targetQ - self.Q)
                self.loss = tf.reduce_mean(self.td_error)
                self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
                self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def evaluate(env, sess, q_max, input_ph, visualize=False, v_func=None):
    if visualize:
        assert (v_func is not None), "Need v_func to visualize the value function"
        env.visualize_on()
        plot_value_function(sess, q_max, v_func, input_ph, env)
    else:
        env.visualize_off()
    succ_epi = 0
    acc_rew = 0
    for _epi in range(1, 101):
        _s = env.reset()  # .flatten()
        _d = False
        while not _d:
            _a = sess.run(q_max, feed_dict={input_ph: [_s]})[0]
            _s1, _r, _d = env.next(np.argmax(_a))
            _s = _s1  # .flatten()
            acc_rew += _r
        succ_epi += 1 if _r == 1 else 0
    succ_epi = succ_epi * 100.0 / _epi

    return succ_epi, acc_rew


def run_demo_agent(path, replay_buffer, demo_trans_size):
    envs = {
        1: ('FrozenLake-v0', None),
        2: ('FrozenLakeNonskid4x4-v0', None),
        3: ('FrozenLakeNonskid8x8-v0', None),
        4: ('CorridorSmall-v1', CorridorEnv),
        5: ('CorridorSmall-v2', CorridorEnv),
        6: ('CorridorActionTest-v0', CorridorEnv),
        7: ('CorridorActionTest-v1', ComplexActionSetCorridorEnv),
        8: ('CorridorBig-v0', CorridorEnv),
        9: ('CorridorFLNonSkid-v1', CorridorEnv)
    }
    env_id = 9
    env = GymEnvironment(-1, envs[env_id][0], 3, env_class=envs[env_id][1], visualize=False, agent_history_length=1)
    num_actions = env.env.action_space.n
    input_shape = list(env.env.observation_space.shape)
    hiddens = [50, 50]

    # with tf.device(''/cpu:0''):
    # with tf.name_scope('demo_agent'):
    # tf.reset_default_graph()
    mainQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)
    targetQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)  # not used but needed to restore

    init = tf.global_variables_initializer()

    # var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='corridor_demo_agent')
    var_list = [var for var in tf.trainable_variables() if 'fully_connected' in var.name]
    # print(var_list)
    saver = tf.train.Saver(var_list=var_list)
    state = env.get_initial_state()
    state_shape = state.shape
    visited_init_states = [np.argmax(state.flatten())]
    with tf.Session() as sess:
        sess.run(init)
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        # print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        for demo_data_counter in range(1, demo_trans_size + 1):
            action = sess.run(mainQN.predict, feed_dict={mainQN.input: [state.flatten()]})[0]
            action = np.eye(num_actions)[action]
            next_state, reward, done = env.next(action)
            # next_state = next_state.flatten()
            # print(state, action, reward, next_state, done)
            replay_buffer.add(state, action, reward, next_state, done)
            np.copyto(state, next_state)
            if ((demo_data_counter % 1000 == 0 and demo_data_counter != 0)
                or (demo_data_counter == demo_trans_size)):
                print("Added {} of {} demo transitions".format(str(demo_data_counter), str(
                    demo_trans_size)))  # + str() + " of " + str(args.demo_trans_size) + " demo transitions")
            if done:
                while True:
                    state = env.get_initial_state()
                    if len(visited_init_states) > 52:  # All possible initial states have been already visited
                        break
                    id = np.argmax(state.flatten())
                    if not id in visited_init_states:
                        visited_init_states.append(id)
                        break
                        # print(len(visited_init_states))
                        # print("ADDINGTO BUFFER:", state.shape, one_hot_action.shape)
                        # return state_shape


def train_agent():
    batch_size = 32  # How many experiences to use for each training step.
    update_freq = 4  # How often to perform a training step.
    y = .99  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0  # 0.1 #Final chance of random action
    max_epLength = 100  # The max allowed length of our episode.
    load_model = False  # Whether to load a saved model.
    tau = 0.001  # Rate to update target network toward primary network
    hiddens = [50, 50]

    envs = {
        1: ('FrozenLake-v0', None),
        2: ('FrozenLakeNonskid4x4-v0', None),
        3: ('FrozenLakeNonskid8x8-v0', None),
        4: ('CorridorSmall-v1', CorridorEnv),
        5: ('CorridorSmall-v2', CorridorEnv),
        6: ('CorridorActionTest-v0', CorridorEnv),
        7: ('CorridorActionTest-v1', ComplexActionSetCorridorEnv),
        8: ('CorridorBig-v0', CorridorEnv),
        9: ('CorridorFLNonSkid-v1', CorridorEnv)
    }
    env_id = 9
    env = GymEnvironment(envs[env_id][0], 3, env_class=envs[env_id][1], visualize=False, agent_history_length=1)
    env2 = GymEnvironment(envs[env_id][0], 4, env_class=envs[env_id][1], visualize=False, agent_history_length=1)
    num_actions = env.env.action_space.n
    input_shape = list(env.env.observation_space.shape)

    if env_id in [1, 3, 6, 7]:
        num_episodes = 65000  # How many episodes of game environment to train network with.
        annealing_steps = 200000.  # How many steps of training to reduce startE to endE.
    else:
        num_episodes = 15000  # How many episodes of game environment to train network with.
        annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
    pre_train_steps = 10000  # How many steps of random actions before training begins.

    succ_threshold = 82 if env_id in [1] else 100

    tf.reset_default_graph()
    mainQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)
    targetQN = Qnetwork(input_shape, hiddens, num_actions, layer_norm=False)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    myBuffer = experience_buffer()

    # Set the rate of random action decrease.
    e = startE
    stepDrop = (startE - endE) / annealing_steps

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0
    acc_rew = 0

    k = 100

    # Make a path for our model to be saved in.
    path = "models/" + envs[env_id][0]  # The path to save our model to.
    if not os.path.exists(path):
        os.makedirs(path)

    finish = False
    with tf.Session() as sess:
        sess.run(init)
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        succ_epi = 0
        for epi in range(1, num_episodes + 1):
            if finish: break
            episodeBuffer = experience_buffer()
            # Reset environment and get first new observation
            s = env.reset().flatten()
            d = False
            rAll = 0
            j = 0
            # The Q-Network
            while (j < max_epLength) and (
            not d):  # If the agent takes longer than 100 moves to reach either of the blocks, end the trial.
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0, 4)
                else:
                    # if e == 0: print(a)
                    a = sess.run(mainQN.predict, feed_dict={mainQN.input: [s]})[0]
                s1, r, d = env.step(a)
                s1 = s1.flatten()
                total_steps += 1
                episodeBuffer.add(
                    np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                        e = max(e, 0)

                    # if e == 0:  env.visualize_on()

                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict, feed_dict={mainQN.input: np.vstack(trainBatch[:, 3])})
                        Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.input: np.vstack(trainBatch[:, 3])})
                        end_multiplier = -(trainBatch[:, 4] - 1)
                        doubleQ = Q2[range(batch_size), Q1]
                        targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, \
                                     feed_dict={mainQN.input: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                                mainQN.actions: trainBatch[:, 1]})

                        updateTarget(targetOps, sess)  # Update the target network toward the primary network.
                rAll += r
                s = s1

            if r == 1: succ_epi += 1

            myBuffer.add(episodeBuffer.buffer)
            # jList.append(j)
            rList.append(rAll)
            # Periodically save the model.
            if epi % 1000 == 0:
                if False:
                    # Evaluate
                    _succ_epi, _acc_rew = evaluate(env2, sess, mainQN.predict, mainQN.input)
                    if _succ_epi >= succ_threshold:
                        print("Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi,
                                                                                                         _acc_rew))
                        break
                saver.save(sess, path + '/model-' + str(epi) + '.ckpt')

            if epi % k == 0:
                acc_rew += np.sum(rList)
                print(
                    "Episodes: {}, Steps: {}, Epi. succ. rate: {}%, Avg. rew/epi:{:.2f}%, Acc. reward: {:.2f}, Epsilon: {:.2f}".format(
                        epi, total_steps, succ_epi * 100.0 / k, np.mean(rList) * 100, acc_rew, e))
                rList = []
                succ_epi = 0

                if e == 0:
                    # Evaluate
                    _succ_epi, _acc_rew = evaluate(env2, sess, mainQN.predict, mainQN.input)
                    if _succ_epi >= succ_threshold:
                        print("Evaluation success rate over 100 episodes: {}%; Total reward: {} ".format(_succ_epi,
                                                                                                         _acc_rew))
                        break

        saver.save(sess, path + '/model-' + str(epi) + '.ckpt')
        print("Model saved")
        print("Training succes rate: {}%".format(acc_rew * 100.0 / epi))

        evaluate(env2, sess, mainQN.predict, mainQN.input, visualize=True, v_func=mainQN.Value)