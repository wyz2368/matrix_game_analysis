import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from bargaining.ppo import PPO, Memory

class Agent():
    def __init__(self, env, discount):
        self._strategies = []
        self._discount = discount
        self._env = env

    def utility_function(self, offer, type="linear"):
        if type == "linear":
            return np.sum(offer)
        else:
            raise NotImplementedError

    def training(self):
        ############## Hyperparameters ##############
        # creating environment
        env = self._env
        num_actions = env._offer_dim

        state_dim = num_actions
        action_dim = num_actions + 1

        log_interval = 20  # print avg reward in the interval
        max_episodes = 5000  # max training episodes
        max_timesteps = 20  # max timesteps in one episode

        n_latent_var = 64  # number of variables in hidden layer
        update_timestep = 30  # update policy every n timesteps
        lr = 0.002
        betas = (0.9, 0.999)
        gamma = 1  # discount factor
        K_epochs = 4  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO
        #############################################

        episode_rewards = []
        episode_rewards.append(0.0)

        memory = Memory()
        ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

        # logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0

        reward_list = []
        x_axis = []

        # training loop
        for i_episode in range(1, max_episodes + 1):
            state = env.reset()
            for t in range(max_timesteps):
                timestep += 1
                # Running policy_old:
                action = ppo.policy_old.act(state, memory, env)

                # Top k
                state, reward, done = env.step(action)

                episode_rewards[-1] += reward

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if timestep % update_timestep == 0:
                    ppo.update(memory)
                    memory.clear_memory()
                    timestep = 0

                running_reward += reward

                if done:
                    episode_rewards.append(0.0)
                    break

            avg_length += t

            # logging
            if i_episode % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                running_reward = int((running_reward / log_interval))

                print('Episode {} \t avg length: {} \t reward: {} \t mean_reward: {}'.format(i_episode, avg_length,
                                                                                             running_reward,
                                                                                             episode_rewards[-2]))

                x_axis.append(i_episode)
                reward_list.append(running_reward)
                running_reward = 0
                avg_length = 0

        return ppo




