import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from algo.utils import ReplayBuffer, SoftQNetwork, SoftQNetworkWithAttention, PolicyNetwork


class SAC():
    def __init__(self, state_dim, action_dim, action_range):
        self.num_training = 0
        hidden_dim = 512 #512

        self.replay_buffer_size = 20000
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)


        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        # 2.5e-4-->2e-4-->2e-3
        soft_q_lr = 2e-4
        policy_lr = 2e-4 #2e-3
        alpha_lr = 2e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.action_dim =1
        self.action_range = 1

    #0.9999-->0.99
    def train(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-3, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        # 关键修正：确保动作维度正确
        action = action.reshape(-1, self.action_dim)  # 强制转换为 [batch_size, action_dim]
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem

        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return alpha_loss, q_value_loss1, q_value_loss2, policy_loss

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + 'q1.pth')
        torch.save(self.soft_q_net2.state_dict(), path + 'q2.pth')
        torch.save(self.policy_net.state_dict(), path + 'policy.pth')

        print('=============The SAC model is saved=============')


    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + 'q1.pth'))
        self.soft_q_net2.load_state_dict(torch.load(path + 'q2.pth'))
        self.policy_net.load_state_dict(torch.load(path + 'policy.pth'))
        self.target_soft_q_net1 = deepcopy(self.soft_q_net1)
        self.target_soft_q_net2 = deepcopy(self.soft_q_net2)

    def get_bc_action(self, state):
        """专门用于BC策略的动作生成"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, _ = self.policy_net(state)
        action = self.action_range * torch.tanh(mean)
        return action.detach().cpu().numpy().flatten()  # 确保返回1D数组 [action_dim]
