import numpy as np
import tensorflow as tf
import random
from collections import deque
import config
import os
import datetime  # 【TensorBoard】新增


class DQNAgent:
    def __init__(self):
        self.state_dim = config.NODE_NUM * 2
        self.action_dim = config.NODE_NUM

        self.memory = deque(maxlen=config.MEMORY_CAPACITY)
        self.epsilon = config.EPSILON_START

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # 【TensorBoard】
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/DQN_' + current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_dim, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(1, config.NODE_NUM - 1)

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor)
        action = np.argmax(q_values[0])
        if action == 0:
            action = random.randint(1, config.NODE_NUM - 1)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < config.BATCH_SIZE:
            return None  # 【TensorBoard】没有训练时返回 None

        minibatch = random.sample(self.memory, config.BATCH_SIZE)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(config.BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + config.GAMMA * np.amax(target_next[i])

        # 【TensorBoard】获取训练历史，提取 Loss
        history = self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > config.EPSILON_MIN:
            self.epsilon *= config.EPSILON_DECAY

        # 返回当前的 Loss 值
        return history.history['loss'][0]

    def save(self, filepath):
        self.model.save(filepath)
        print(f"模型已保存至: {filepath}")

    def load(self, filepath):
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            self.update_target_model()
            self.epsilon = 0.0
            print(f"模型已加载: {filepath}")
        else:
            print(f"错误：找不到模型文件 {filepath}")