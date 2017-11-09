import tensorflow as tf
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Flatten, Input, Lambda, Activation
from tensorflow.contrib.layers import layer_norm
import numpy as np

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class SimpleQNetwork(object):
    def __init__(self, conf, learning_network=None):
        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.clip_loss_delta = conf['clip_loss_delta']
        self.clip_norm = conf['clip_norm']
        self.clip_norm_type = conf['clip_norm_type']
        self.device = conf['device']
        self.arch = conf['arch']

        self.input_shape = 64
        # self.hiddens = [50, 50]
        self.num_actions = 4
        self.dueling_type = 'avg'
        self.layer_norm = False

        with tf.device(self.device):
            with tf.name_scope(self.name): # Obs! Keras does not propagate name_scope to the variables it creates internally
                self.input_ph = tf.placeholder(shape=(None, 64, 1), dtype=tf.float32, name='input')
                input = Input(tensor=self.input_ph, name=self.name)
                # input = Input(shape= (64, 1,))
                out = Flatten(name=self.name)(input)
                for nodes in [50, 50]:  # conf['mlp_hiddens']:
                    out = Dense(nodes, name=self.name)(out)
                    if False:  # conf['layer_norm']:
                        # out = Lambda(lambda x: layer_norm(x))(out, center=True, scale=True)
                        out = Lambda(layer_norm, center=True, scale=True, name=self.name)(out)
                    out = Activation('relu', name=self.name)(out)

                self.advantage = Dense(self.num_actions, name=self.name)(out)
                self.value = Dense(1, name=self.name)(out)

                # self.w1, self.b1, fc1 = fc('fc1', self.input_ph, 50, activation="relu")
                # self.w2, self.b2, fc2 = fc('fc2', fc1, 50, activation="relu")
                # #
                # self.w3_a, self.b3_a, self.advantage = fc('adv_f', fc2, self.num_actions, activation="linear")
                # self.w3_v, self.b3_v, self.value = fc('value_f', fc2, 1, activation="linear")

                if self.dueling_type == 'avg':
                    def avg_dueling(value, advantage):
                        return tf.add(value, tf.subtract(advantage,
                                                         tf.reduce_mean(
                                                             advantage,
                                                             axis=1,
                                                             keep_dims=True)))

                    self.output_layer_q = Lambda(avg_dueling, arguments={'advantage': self.advantage})(self.value)
                elif self.dueling_type == 'max':
                    def max_dueling(value, advantage):
                        return tf.add(value, tf.subtract(advantage,
                                                         tf.reduce_max(
                                                             advantage,
                                                             axis=1,
                                                             keep_dims=True)))

                    self.output_layer_q = Lambda(max_dueling, arguments={'advantage': self.advantage}, name=self.name)(self.value)

                else:
                    assert False, "dueling_type must be one of {'avg','max'}"

                #self.model = Model(input, self.output_layer_q)

                # self.params = [self.w1, self.b1,
                #               self.w2, self.b2,
                #               self.w3_a, self.b3_a,
                #               self.w3_v, self.b3_v]
                self.params = [v for v in tf.trainable_variables() if self.name in v.name] #self.model.trainable_weights does not seem to work properly


                # NB. Both 1-step and n-step targets can be part of the batch.
                # This means that for every state-action pair, there will be
                # two target values
                # If so, then the below loss function will work as is
                if "value_learning" in self.name:  # learning network
                    self.selected_action_ph = tf.placeholder("int32", [None],#, self.num_actions],
                                                             name="selected_action")
                    self.selected_action = tf.one_hot(self.selected_action_ph, self.num_actions, dtype=tf.float32)

                    self.target_ph = tf.placeholder("float32", [None], name='target')

                    self.output_selected_action = tf.reduce_sum(
                        tf.multiply(self.output_layer_q, self.selected_action),
                        reduction_indices=1)


                    # TD loss (Huber loss)
                    delta = self.clip_loss_delta
                    self.td_error = tf.subtract(self.target_ph, self.output_selected_action)
                    self.td_loss = tf.reduce_mean(
                                tf.where(tf.abs(self.td_error) < delta,
                                       tf.square(self.td_error) * 0.5,
                                       delta * (tf.abs(self.td_error) - 0.5 * delta)))

                    # if self.clip_loss_delta > 0:
                    #     quadratic_part = tf.minimum(tf.abs(diff),
                    #                                 tf.constant(self.clip_loss_delta))
                    #     linear_part = tf.sub(tf.abs(diff), quadratic_part)
                    #     td_loss = tf.add(tf.nn.l2_loss(quadratic_part),
                    #                      tf.mul(tf.constant(self.clip_loss_delta), linear_part))
                    # else:
                    #     #td_loss = tf.nn.l2_loss(diff)
                    #     td_loss = tf.reduce_mean(tf.square(diff))


                    # Supervised Large Margin loss
                    self.mask_margin_loss = tf.placeholder(
                        "float32", None, name='mask_margin_loss') # 1. if demo data, else 0.
                    inverted_one_hot = tf.ones_like(self.selected_action) - self.selected_action
                    expert_margin = self.output_layer_q + inverted_one_hot * conf['expert_margin']
                    expert_target = tf.reduce_max(expert_margin, axis=-1)
                    # margin_loss = tf.reduce_mean(tf.square(expert_target - output_selected_action))
                    self.margin_loss = tf.reduce_mean(expert_target - self.output_selected_action)
                    wt_margin_loss = self.margin_loss * self.mask_margin_loss * conf['margin_loss_coeff']

                    # Regularisation loss
                    weights = [p for p in self.params if ("LayerNorm" not in p.name and "bias" not in p.name)]
                    #weights = [p for p in self.params if "LayerNorm" not in p.name]
                    self.reg_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])
                    wt_reg_loss = self.reg_loss * conf['L2_reg_coeff']

                    self.loss = self.td_loss #+ wt_margin_loss + wt_reg_loss

                elif "value_target" in self.name:
                    if conf['continuous_target_update']:
                        assert learning_network is not None, "Need to pass the learning network as argument when creating the target network"
                        tau = tf.constant(conf['target_update_tau'], dtype=np.float32)
                        self.continuous_sync_nets = []
                        for i in range(len(learning_network.params)):
                            self.continuous_sync_nets.append(
                                self.params[i].assign(
                                    tf.multiply(learning_network.params[i].value(), tau) +
                                    tf.multiply(self.params[i], tf.subtract(tf.constant(1.0),tau))))
                    else:
                        self.params_ph = []
                        for p in self.params:
                            self.params_ph.append(tf.placeholder(tf.float32,
                                                                 shape=p.get_shape(),
                                                                 name='params_to_sync'))

                        self.discrete_sync_nets = []
                        for i in range(len(self.params)):
                            self.discrete_sync_nets.append(
                                self.params[i].assign(self.params_ph[i]))

    def get_params(self, session):
        with tf.device(self.device):
            with tf.name_scope(self.name):
                param_values = session.run(self.params)
                return param_values

    def set_params(self, feed_dict, session):
        with tf.device(self.device):
            with tf.name_scope(self.name):
                session.run(self.discrete_sync_nets, feed_dict=feed_dict)

    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device(self.device):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            if path is None:
                # We start from scartch. All the tensorflow graph variables have been already initialized before
                # coming here. Here we just synchronize the learning and target networks
                logging.info('Initialized all variables.')
                #session.run(tf.global_variables_initializer())
            else:
                #session.run(tf.global_variables_initializer())
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-') + 1:])
                logging.info('Restored network variables from previous run')
        return last_saving_step

