import tensorflow as tf
import numpy as np
import math

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations, 
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """  
        self.env = env
        self.sess = sess
        with tf.variable_scope("nn_model", reuse=tf.AUTO_REUSE):
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            self.input_ = tf.placeholder(tf.float32, [None,obs_dim + act_dim], name = "dyn_mod_input")
            self.output_ = tf.placeholder(tf.float32, [None,obs_dim], name = "dyn_mod_output")
            self.normalization = normalization
            self.epoches = iterations
            self.batch_size = batch_size

            self.model_ = build_mlp(self.input_, obs_dim, "dyn_mod",  n_layers, size, activation, output_activation)
            self.loss = tf.losses.mean_squared_error(self.output_, self.model_)
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

            self.opt = self.optimizer.minimize(self.loss)

            self.mean_obs = np.zeros(env.observation_space.shape)
            self.mean_action = np.zeros(env.action_space.shape)
            self.mean_delta = np.zeros(env.observation_space.shape)
            self.std_obs = np.ones(env.observation_space.shape)
            self.std_action = np.ones(env.action_space.shape)
            self.std_delta = np.ones(env.observation_space.shape)
            
            self.sess = sess
        

    def fit(self, data, verbose=False):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        """YOUR CODE HERE """
        means_stds = self.normalization(data)
        self.mean_obs = means_stds['mean_obs']
        self.mean_delta = means_stds['mean_deltas']
        self.mean_action = means_stds['mean_actions']
        self.std_obs = means_stds['std_obs']
        self.std_delta = means_stds['std_deltas']
        self.std_action = means_stds['std_actions']
        
        obs_normalized = (data['observations'] - self.mean_obs)/self.std_obs
        delta_normalized = ((data['next_observations'] - data['observations']) - self.mean_delta)/self.std_delta
        action_normalized = (data['actions'] - self.mean_action)/self.std_action
        
        data_size = len(obs_normalized)
        for i in range(self.epoches):
            index = np.arange(data_size)
            np.random.shuffle(index)
            obs_normalized = obs_normalized[index]
            delta_normalized = delta_normalized[index]
            action_normalized = action_normalized[index]
            
            bs = self.batch_size
            
            for i in range(math.floor(data_size/bs)):
                obs_batch = obs_normalized[i*bs: (i+1)*bs]
                delta_batch = delta_normalized[i*bs: (i+1)*bs]
                action_batch = action_normalized[i*bs: (i+1)*bs]
                inputs_batch = np.hstack([obs_batch, action_batch])
                
                _, loss = self.sess.run([self.opt, self.loss], feed_dict={self.input_: inputs_batch, self.output_: delta_batch})
                if verbose and i and i%10 == 9:
                    print (loss)

        
        
        
    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        states = (states - self.mean_obs)/self.std_obs
        actions = (actions - self.mean_action)/self.std_action
        inputs = np.hstack([states, actions])
        
        delta = self.sess.run(self.model_, feed_dict={self.input_: inputs})
        delta = (delta * self.std_delta) + self.mean_delta
        next_states = states + delta
        return next_states