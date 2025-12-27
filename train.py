# train.py
import numpy as np
import random
import tensorflow as tf
import shutil
import os
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent

# ================= 训练控制配置 =================
RESUME_TRAINING = False  # 是否从断点恢复训练
# 若 RESUME_TRAINING 为 True，请指定下方路径
CHECKPOINT_PATH = "logs/DQN_2025xxxx-xxxx/checkpoint_5000.h5"


def train():
    env = NetworkEnvironment()
    agent = DQNAgent()

    # --- 断点续训逻辑 ---
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"=== 正在从断点恢复: {CHECKPOINT_PATH} ===")
        agent.load(CHECKPOINT_PATH)
        agent.epsilon = 0.1  # 恢复训练时给予少量探索，避免陷入局部最优
        print(f"=== Epsilon 已重置为: {agent.epsilon} (保持探索) ===")
    else:
        print("=== 开始新的训练 (从零开始) ===")

    active_tasks = []

    # 统计指标
    step_rewards, step_losses, step_costs = [], [], []
    total_attempts = 0
    success_count = 0

    print(f"=== 开始训练 (总步数: {config.MAX_STEPS}) ===")
    print(f"=== TensorBoard 日志目录: {agent.log_dir} ===")

    for time_step in range(config.MAX_STEPS):

        # 1. 资源释放 (遍历检查已过期的任务)
        for i in range(len(active_tasks) - 1, -1, -1):
            task = active_tasks[i]
            if task['end_time'] <= time_step:
                env.release_task_resources(task)
                active_tasks.pop(i)

        # 2. 批量任务生成 (泊松分布模拟真实流量)
        current_batch_size = np.random.poisson(config.TASK_ARRIVAL_RATE)
        current_batch_size = min(current_batch_size, 10)  # 限制峰值防止过载

        # 处理本批次的所有任务
        for _ in range(current_batch_size):
            total_attempts += 1

            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            # 获取状态 -> 决策 -> 执行
            state = env.get_global_state(0, task=new_task)
            action_node = agent.act(state)
            next_state, reward, done, info = env.step(action_node, new_task)

            # 存储经验并训练
            agent.remember(state, action_node, reward, next_state, done)
            loss = agent.replay()

            # 记录数据
            step_rewards.append(reward)
            if loss is not None:
                step_losses.append(loss)

            if info['status'] == 'Success':
                success_count += 1
                step_costs.append(info['cost'])
                active_tasks.append({
                    'end_time': time_step + new_task['duration'],
                    'node_id': action_node,
                    'cpu': new_task['cpu'],
                    'bw': new_task['bw'],
                    'path': info['path']
                })

        # 3. 定期保存模型 Checkpoint (每2000步)
        if time_step % 2000 == 0 and time_step > 0:
            ckpt_path = os.path.join(agent.log_dir, f"checkpoint_{time_step}.h5")
            agent.save(ckpt_path)

        # 4. TensorBoard 日志与 Target 网络更新 (每100步)
        if time_step % 100 == 0 and time_step > 0:
            agent.update_target_model()

            avg_reward = np.mean(step_rewards) if step_rewards else 0
            avg_loss = np.mean(step_losses) if step_losses else 0
            avg_cost = np.mean(step_costs) if step_costs else 0
            current_success_rate = success_count / total_attempts if total_attempts > 0 else 0

            with agent.summary_writer.as_default():
                tf.summary.scalar('Main/Average_Reward', avg_reward, step=time_step)
                tf.summary.scalar('Main/Loss', avg_loss, step=time_step)
                tf.summary.scalar('Performance/Success_Rate', current_success_rate, step=time_step)
                tf.summary.scalar('Performance/Average_Cost', avg_cost, step=time_step)
                tf.summary.scalar('Parameters/Epsilon', agent.epsilon, step=time_step)

            print(
                f"Step {time_step} | Reward: {avg_reward:.2f} | Loss: {avg_loss:.4f} | Success: {current_success_rate:.2%}")

            # 重置局部统计
            step_rewards, step_losses, step_costs = [], [], []

    # 5. 训练结束保存最终模型
    print("-" * 50)
    save_path = os.path.join(agent.log_dir, "dqn_trained_model.h5")
    agent.save(save_path)

    # 备份配置文件以供复现
    shutil.copy('config.py', agent.log_dir)

    print(f"训练完成！模型保存路径: {save_path}")


if __name__ == "__main__":
    train()