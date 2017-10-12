from networks import *


class QNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(QNetwork, self).__init__(conf)

        # self.entropy_regularisation_strength = conf['entropy_regularisation_strength']

        with tf.device(self.device):
            with tf.name_scope(self.name):

                # self.target_ph = tf.placeholder(
                #     "float32", [None], name='target')
                
                # # 1. if demo data, else 0.
                # self.mask_margin_loss = tf.placeholder(
                #     "float32", None, name='mask_margin_loss')

                # Final Q value layer
                self.wf, self.bf, self.output_layer_q = fc('q_output', self.output, self.num_actions, activation="linear")
                
                if self.arch == 'NIPS':
                    self.params = [self.w1, self.b1, 
                                    self.w2, self.b2, 
                                    self.w3, self.b3, 
                                    self.wf, self.bf]                
                else:
                    self.params = [self.w1, self.b1, 
                                    self.w2, self.b2,
                                    self.w3, self.b3, 
                                    self.w4, self.b4, 
                                    self.wf, self.bf]                

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
                    if self.arch == 'NIPS':
                        trainable_weights = [self.w1, 
                                                self.w2, 
                                                self.w3, 
                                                self.wf]
                    else:
                        trainable_weights = [self.w1, 
                                                self.w2, 
                                                self.w3, 
                                                self.w4, 
                                                self.wf]

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


class NIPSQNetwork(QNetwork, NIPSNetwork):
    pass


class NatureQNetwork(QNetwork, NatureNetwork):
    pass
