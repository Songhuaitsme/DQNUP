import numpy as np
import tensorflow as tf
import random
import os
import datetime
from collections import deque
import config

class DQNAgent:
    def __init__(self):
        self.state_dim = config.INPUT_DIM
        self.action_dim = config.NODE_NUM

        self.memory = deque(maxlen=config.MEMORY_CAPACITY)
        self.epsilon = config.EPSILON_START

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # TensorBoard 日志设置
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join('logs', f'DQN_{current_time}')
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def _build_model(self):
        """构建全连接神经网络"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=config.INPUT_DIM, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        # 使用 Huber Loss 提高训练稳定性
        model.compile(loss=tf.keras.losses.Huber(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
        return model

    def update_target_model(self):
        """同步 Target 网络权重"""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """Epsilon-Greedy 策略选择动作"""
        # 随机探索
        if np.random.rand() <= self.epsilon:
            return random.randint(1, config.NODE_NUM - 1)

        # 模型预测
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor)
        action = np.argmax(q_values[0])

        # 强制修正：如果模型选择了节点0 (调度器)，则随机重选有效节点
        if action == 0:
            action = random.randint(1, config.NODE_NUM - 1)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < config.BATCH_SIZE:
            return None

        minibatch = random.sample(self.memory, config.BATCH_SIZE)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # 预测当前Q值和目标Q值
        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(config.BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + config.GAMMA * np.amax(target_next[i])

        # 训练一步
        history = self.model.fit(states, targets, epochs=1, verbose=0)

        # 衰减 Epsilon
        if self.epsilon > config.EPSILON_MIN:
            self.epsilon *= config.EPSILON_DECAY

        return history.history['loss'][0]

    def save(self, filepath):
        self.model.save(filepath)
        print(f"模型已保存至: {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            self.update_target_model()
            self.epsilon = 0.0  # 加载时默认关闭探索，若需训练需外部手动重置
            print(f"模型已加载: {filepath}")
        else:
            print(f"错误：找不到模型文件 {filepath}")