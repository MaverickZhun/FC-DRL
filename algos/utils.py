import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple
Experience = namedtuple('Experience',('agent_id','step', 'state', 'action', 'reward', 'next_state', 'done'))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # self.buffer[self.position] =  (state, action, reward, next_state, done)
        # 确保动作存储为二维数组
        action = np.array(action).reshape(-1)  # 转换为1D数组 (即使action_dim=1)，新增：
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        # 转换为正确形状,新增：
        action = action.reshape(-1, 1)  # [B, action_dim=1]
        return state, action, reward, next_state, done

    def sample_all(self, batch_size):
        state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))  # stack for each element
        return state, action, reward, next_state, done

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)


class BCReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action = map(np.stack, zip(*batch))
        return state, action

    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # the dim 0 is number of samples
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class SoftQNetworkWithAttention(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, num_heads=2, init_w=3e-3, attention_weight=0.05):
        super(SoftQNetworkWithAttention, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_weight = attention_weight  # 控制自注意力的影响

        # Linear layers for input processing
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        # Self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

        # Final output layer
        self.linear4 = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # Concatenate state and action as the input for the network
        x = torch.cat([state, action], dim=1)  # (batch_size, state+action_dim)

        # Pass through the first few layers
        x = F.relu(self.linear1(x))  # (batch_size, hidden_size)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        # Reshape for self-attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_size) to fit the expected input format

        # Apply self-attention (not too much influence on the final result)
        attn_output, _ = self.attn(x, x, x)  # (batch_size, 1, hidden_size)

        # Remove the sequence dimension, return to (batch_size, hidden_size)
        x = attn_output.squeeze(1)  # (batch_size, hidden_size)

        # Apply the attention weight to reduce its effect
        x = (1 - self.attention_weight) * F.relu(self.linear3(x)) + self.attention_weight * x

        # Final output layer
        q_value = self.linear4(x)  # (batch_size, 1)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0

        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action

    def sample_action(self, ):
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range * a.numpy()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
