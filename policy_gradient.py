import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    '''
    A class that represents the policy network used as
    a function approximator between the current state
    of the environment to the probabilities of taking
    each action. This is also known as a policy.
    '''

    def __init__(self, state_space, action_space, hidden_size):
        '''
        Arguments:
            StateSpace: the state_space object of the environment
            ActionSpace: the action_space object of the environment
            hidden_size: the number of neurons to have in the neural network.
        '''
        super().__init__()

        # Retrieves the state space dimension.
        input_dim = state_space.high.shape[0]

        # Retrieves the action space dimension.
        output_dim = action_space.n

        # Define the neural network.
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax()
        )

    def forward(self, state):
        '''
        Given a state, performs a forward pass
        of the neural network.

        Arguments:
            state: a state in the environment

        Returns:
            output: the output of the neural network.
        '''

        output = self.fc(state)

        return output


class PGAgent(object):
    '''
    A class that represents an Agent that follows a policy
    derived by a Policy-Gradients reinforcement learning method.
    '''

    def __init__(self, policy_network, gamma):
        '''
        Arguments:
            policy_network: the policy network the agent will follow.
            gamma: the discount factor
        '''
        self.policy_network = policy_network
        self.gamma = gamma

        # Stores the history and reward of the policy over an episode.
        self.policy_history = None
        self.policy_reward = []

    def select_action(self, state):
        '''
        Selects an action by sampling from the policy
        at a given state.

        Arguments:
            state: the current state of the environment

        Returns:
            action: the action to take based on this state
        '''

        # Passes the state through the policy network to get
        # a probability distribution of actions.
        pi_s = self.policy_network(state)

        # Sample an action from this distribution.
        c_action = Categorical(pi_s)
        action = c_action.sample()

        # Calculates the log probabilities of each of the actions
        # in the policy distribution.
        log_action = c_action.log_prob(
            action).view(-1, 1).type(torch.FloatTensor)

        # Caches the log probabilities of each action for later use.
        if self.policy_history is None:
            self.policy_history = log_action

        else:
            self.policy_history = torch.cat([self.policy_history, log_action])

        return action

    def reset_policy(self):
        '''
        Resets the policy network.
        '''
        self.policy_history = None
        self.policy_reward = []

    def update_policy(self, optimiser):
        '''
        Performs a step of gradient descent to optimise
        the weights of the policy network.

        Arguments:
            optimiser: the type of optimiser in use.

        Returns:
            loss: the loss of the episode
            rewards[0]: the total reward of the episode
        '''

        # Keep tracks of the total reward
        R = 0

        # Cache the discounted rewards at each step of the episode
        rewards = []

        # Calculate the discounted reward at each step of the episode.
        for r in self.policy_reward[::-1]:
            r = r.type(torch.FloatTensor)
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Normalise the rewards for stability.
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        # Retrieve the log probabilities of the actions over time.
        log_pi_t = self.policy_history

        # Calculate the loss of the episode
        loss = torch.sum(-torch.mul(log_pi_t, rewards))

        # Perform backpropagation + gradient step.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        return loss, rewards[0]

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")

    state_space = env.observation_space
    action_space = env.action_space
    episodes = 1000

    hidden_size = 16
    discount = 0.99

    policy_network = PolicyNetwork(
        state_space, action_space, hidden_size).double()
    agent = PGAgent(policy_network, discount)

    optimiser = optim.RMSprop(policy_network.parameters())

    episode_rewards = []
    for episode in range(episodes):

        state = torch.from_numpy(env.reset())
        agent.reset_policy()

        total_reward = 0
        done = False
        while not done:
            env.render()
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action.data.numpy())
            total_reward += reward

            state = torch.tensor(state)
            reward = torch.tensor(reward)

            agent.policy_reward.append(reward)

        loss, _ = agent.update_policy(optimiser)

        episode_rewards.append(total_reward)

        if episode % 1 == 0:
            print("Episode: %d -> Loss: %.4f, Reward: %.4f" %
                  (episode, loss, total_reward))
