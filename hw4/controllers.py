import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env
        


    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """        
        """ Note: be careful to batch your simulations through the model for speed """
        N = self.num_simulated_paths
        h = self.horizon
        action_samples = np.random.uniform(self.env.action_space.low, self.env.action_space.high, \
                                            [h, N, *self.env.action_space.shape])
        
        state_samples = [np.tile(state, (N, 1))]
        
        for step in range(h):
            current_state = state_samples[-1]
            current_action = action_samples[step]           
            state_samples.append(self.dyn_model.predict(current_state, current_action))
            
        next_states = np.swapaxes(np.array(state_samples[1:]), 1, 0)
        state_samples = np.swapaxes(np.array(state_samples[:-1]), 1, 0)
        action_samples = np.swapaxes(action_samples, 1, 0)
        
        costs = np.array([trajectory_cost_fn(self.cost_fn, s, a, n) for s, a, n in zip(state_samples, action_samples,
                                                                                       next_states)])
        best_index = costs.argmin()
        best_action = action_samples[best_index][0]
        best_cost = costs[best_index]
        
        return best_action, best_cost
        
        
            
            
            
