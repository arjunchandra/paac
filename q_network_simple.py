import tensorflow as tf
import logging
import numpy as np

def fc(name, _input, output_dim, activation = "relu", init = "torch"):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim],
                           name + '_weights', init = init)
    b = fc_bias_variable([output_dim], input_dim,
                         '' + name + '_biases', init = init)
    out = tf.add(tf.matmul(_input, w), b, name= name + '_out')

    if activation == "relu":
        out = tf.nn.relu(out, name='' + name + '_relu')

    return w, b, out

def fc_weight_variable(shape, name, init="torch"):
    if init == "glorot_uniform":
        fan_in = shape[0]
        fan_out = shape[1]
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')

def fc_bias_variable(shape, input_channels, name, init= "torch"):
    if init=="glorot_uniform":
        initial = tf.zeros(shape, dtype='float32')
    else:
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')

class SimpleQNetwork(object):
    def __init__(self, conf):
        self.input_shape = 64
        # self.hiddens = [50, 50] 
        self.num_actions = 4
        self.dueling_type= 'avg', 
        self.layer_norm=False
        
        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.input_ph =  tf.placeholder(shape= (None, 64),dtype=tf.float32, name='input')

                if "local_learning" in self.name:
                    self.selected_action_ph = tf.placeholder("float32", [None, self.num_actions], name="selected_action")

                self.w1, self.b1, fc1 = fc('fc1', self.input_ph, 50, activation="relu")
                self.w2, self.b2, fc2 = fc('fc2', fc1, 50, activation="relu")

                self.w3_a, self.b3_a, self.advantage = fc('adv_f', fc2, self.num_actions, activation="linear")
                self.w3_v, self.w3_v, self.value = fc('value_f', fc2, 1, activation="linear")
            
                # self.Advantage = layers.fully_connected(out, num_outputs=self.num_actions, activation_fn=None)
                # self.Value = layers.fully_connected(out, num_outputs=1, activation_fn=None)
            
                if self.dueling_type == 'avg':
                    self.output_layer_q = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage,axis=1,keep_dims=True))
                elif self.dueling_type == 'max':
                    self.output_layer_q = self.value + tf.subtract(self.advantage,tf.reduce_max(self.advantage,axis=1,keep_dims=True))
                else:
                    assert False, "dueling_type must be one of {'avg','max'}"        
                

                self.params = [self.w1, self.b1, 
                               self.w2, self.b2, 
                               self.w3_a, self.b3_a,
                               self.w3_v, self.b3_v]

                # NB. Both 1-step and n-step targets can be part of the batch.
                # This means that for every state-action pair, there will be
                # two target values 
                # If so, then the below loss function will work as is
                if "local_learning" in self.name: # learning network
                    self.target_ph = tf.placeholder(
                        "float32", [None], name='target')
                    
                    # 1. if demo data, else 0.
                    self.mask_margin_loss = tf.placeholder(
                        "float32", None, name='mask_margin_loss')

                    output_selected_action = tf.reduce_sum(
                        tf.multiply(self.output_layer_q, self.selected_action_ph), 
                        reduction_indices = 1)

                    diff = tf.subtract(self.target_ph, output_selected_action)
 
                    if self.clip_loss_delta > 0:
                        quadratic_part = tf.minimum(tf.abs(diff), 
                                            tf.constant(self.clip_loss_delta))
                        linear_part = tf.sub(tf.abs(diff), quadratic_part)
                        td_loss = tf.add(tf.nn.l2_loss(quadratic_part),
                                        tf.mul(tf.constant(self.clip_loss_delta), linear_part))
                    else:
                        td_loss = tf.nn.l2_loss(diff)
                    
                    # Supervised loss
                    inverted_one_hot = tf.ones_like(self.selected_action_ph) - self.selected_action_ph
                    expert_margin = self.output_layer_q + inverted_one_hot * conf['expert_margin']
                    expert_target = tf.reduce_max(expert_margin, axis=-1)
                    #margin_loss = tf.reduce_mean(tf.square(expert_target - output_selected_action))
                    margin_loss = tf.reduce_mean(expert_target - output_selected_action)
                    wt_margin_loss = margin_loss * self.mask_margin_loss * conf['margin_loss_coeff']

                    # Regularisation loss

                    trainable_weights = [self.w1, 
                                         self.w2, 
                                         self.w3_a, 
                                         self.w3_v]

                    # [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if all(s in v.name for s in ['local_learning', 'weights'])]
                    reg_loss = tf.add_n([tf.nn.l2_loss(w) for w in trainable_weights])
                    wt_reg_loss = reg_loss * conf['L2_reg_coeff']
                    
                    self.loss = td_loss + wt_margin_loss + wt_reg_loss

                if "local_target" in self.name:
                    self.params_ph = []
                    for p in self.params:
                        self.params_ph.append(tf.placeholder(tf.float32, 
                            shape = p.get_shape(),
                            name = 'params_to_sync'))

                    self.sync_net = []
                    for i in range(len(self.params)):
                        self.sync_net.append(
                            self.params[i].assign(self.params_ph[i]))

    def get_params(self, session):
        with tf.device(self.device):
            with tf.name_scope(self.name):
                param_values = session.run(self.params)
                return param_values

    def set_params(self, feed_dict, session):
        with tf.device(self.device):
            with tf.name_scope(self.name):
                session.run(self.sync_net, feed_dict=feed_dict)


    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device(self.device):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            if path is None:
                logging.info('Initializing all variables')
                session.run(tf.global_variables_initializer())
            else:
                logging.info('Restoring network variables from previous run')
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-')+1:])
        return last_saving_step
