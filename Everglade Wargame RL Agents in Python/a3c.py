import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    # print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        

class a3c(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, state_dim, action_dim, has_continuous_action_space):
        super(a3c, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim

        self.affine1 = nn.Linear(105, 1)

        # actor's layer
        self.action_head = nn.Linear(1, 128)

        # critic's layer
        self.value_head = nn.Linear(1, 128)
        
        self.buffer = RolloutBuffer()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
            
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling A3C::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        actions = []
        action_log_probs = []
        
        for _ in range(14):
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            actions.append(action.detach())
            action_log_probs.append(action_logprob.detach())
        
        return actions, action_log_probs

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
        
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

state_dim=105
action_dim=14
has_continuous_action_space=False
action_std_init=0.6
policy = a3c(state_dim, action_dim, has_continuous_action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()