#!/usr/bin/env python3
# UTF-8
import gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from tensorboardX import SummaryWriter

from model import DQN


class DQNAgent:
    """
        初始化
        @:param env_id : gym环境id
    """

    def __init__(self, env_id, config):
        # gym
        self._env_id = env_id
        self._env = gym.make(env_id)
        self._state_size = self._env.observation_space.shape[0]
        self._action_size = self._env.action_space.n
        # 参数
        self._gamma = config.gamma
        self._learning_rate = config.lr
        self._reward_boundary = config.reward_boundary
        self._device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
        # model
        self._model = DQN(self._state_size, self._action_size).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        # 经验池
        self._replay_buffer = deque(maxlen=config.buffer_size)
        self._mini_batch = config.mini_batch
        # epsilon
        self._epsilon = config.epsilon
        self._epsilon_min = config.epsilon_min
        self._epsilon_decay = config.epsilon_decay

    """
        将observation放入双向队列中，队列满时自动删除最旧的元素
    """

    def remember(self, state, action, next_state, reward, done):
        self._replay_buffer.append((state, action, next_state, reward, done))

        # epsilon幂指数下降
        if len(self._replay_buffer) > self._mini_batch:
            if self._epsilon > self._epsilon_min:
                self._epsilon *= self._epsilon_decay
        pass

    """
        epsilon-greedy action
    """

    def act(self, state):
        # 类似模拟退火，random返回[0,1]
        if np.random.random() <= self._epsilon:
            return random.randrange(self._action_size)
        else:
            # numpy转成tensor，unsqueeze在下标0处新增一个维度
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self._device)
            # 模型预测
            predict = self._model(state)
            # max在第1维处取最大，[1]为下标，[0]为值， [512*2]-> [521]
            return predict.max(1)[1].item()
        pass

    """
        训练
        1、从双向队列中采样mini_batch
        2、预测next_state
        3、更新优化器
    """

    def replay(self):
        if len(self._replay_buffer) < self._mini_batch:
            return
        # 1、从双向队列中采样mini_batch
        mini_batch = random.sample(self._replay_buffer, self._mini_batch)

        # 载入方式一
        # state = np.zeros((self._mini_batch, self._state_size))
        # next_state = np.zeros((self._mini_batch, self._state_size))
        # action, reward, done = [], [], []
        #
        # for i in range(self._mini_batch):
        #     state[i] = mini_batch[i][0]
        #     action.append(mini_batch[i][1])
        #     next_state[i] = mini_batch[i][2]
        #     reward.append(mini_batch[i][3])
        #     done.append(mini_batch[i][4])

        # 载入方式二
        state, action, next_state, reward, done = zip(*mini_batch)
        state = torch.tensor(state, dtype=torch.float).to(self._device)
        action = torch.tensor(action, dtype=torch.long).to(self._device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self._device)
        reward = torch.tensor(reward, dtype=torch.float).to(self._device)
        done = torch.tensor(done, dtype=torch.float).to(self._device)

        # 2、预测next_state
        q_target = reward + \
                   self._gamma * self._model(next_state).to(self._device).max(1)[0] * (1 - done)

        q_values = self._model(state).to(self._device).gather(1, action.unsqueeze(1)).squeeze(1)
        loss_func = nn.MSELoss()
        loss = loss_func(q_values, q_target)
        # loss = (q_values - q_target.detach()).pow(2).mean()

        # 3、更新优化器
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    """
        1、渲染gym环境开始交互
        2、训练模型
    """

    def training(self):
        writer = SummaryWriter(comment="-train-" + self._env_id)
        print(self._model)

        # 参数
        frame_index = 0
        episode_index = 1
        best_mean_reward = None
        mean_reward = 0
        total_rewards = []

        while mean_reward < self._reward_boundary:

            state = self._env.reset()
            # 一轮结束，reward置零
            episode_reward = 0

            while True:
                # 1、渲染gym环境开始交互
                self._env.render()

                # 选择action进行交互
                action = self.act(state)
                next_state, reward, done, _ = self._env.step(action)
                self.remember(state, action, next_state, reward, done)
                state = next_state
                frame_index += 1
                episode_reward += reward

                # 2、训练模型
                loss = self.replay()

                # 游戏结束，开始训练模型
                if done:
                    if loss is not None:
                        print("episode: %4d, frames: %5d, reward: %5f, loss: %4f, epsilon: %4f" % (
                            episode_index, frame_index, np.mean(total_rewards[-10:]), loss, self._epsilon))

                    episode_index += 1
                    total_rewards.append(episode_reward)
                    mean_reward = np.mean(total_rewards[-10:])

                    writer.add_scalar("epsilon", self._epsilon, frame_index)
                    writer.add_scalar("episode_reward", episode_reward, frame_index)
                    writer.add_scalar("mean_reward", mean_reward, frame_index)
                    if best_mean_reward is None or best_mean_reward < mean_reward:
                        torch.save(self._model.state_dict(), "training-best.dat")
                    break

        self._env.close()
        pass

    def test(self, model_path):
        if model_path is None:
            return
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

        total_rewards = []

        for episode_index in range(10):
            episode_reward = 0
            done = False
            state = self._env.reset()

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self._env.step(action)

                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)
            print("episode: %4d, reward: %5f" % (episode_index, np.mean(total_rewards[-10:])))
