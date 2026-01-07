# train.py
'''并行分配失败率极低'''
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

        # ... (前文代码: 资源释放逻辑保持不变)
        # 2. 批量任务生成 (泊松分布)
        current_batch_size = np.random.poisson(config.TASK_ARRIVAL_RATE)
        current_batch_size = min(current_batch_size, 10)  # 限制峰值

        # ==========================================
        # 新增逻辑：并发模拟 (Concurrent Simulation)
        # ==========================================

        # 临时存储本批次的决策，用于稍后统一执行
        # 格式: (task_dict, state_at_decision_time, action_chosen)
        batch_decisions = []

        # --- 第一阶段：并发决策 (Decision Phase) ---
        # 所有任务看到的都是本时间步开始时的“快照”状态
        # 注意：这里我们不调用 env.step，资源状态不会改变
        for _ in range(current_batch_size):
            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            # 获取状态 (此时获取的是同一时刻的资源状态，除非你手动刷新)
            state = env.get_global_state(0, task=new_task)

            # 智能体决策
            action_node = agent.act(state)

            # 暂存决策，暂不执行
            batch_decisions.append((new_task, state, action_node))

        # --- 第二阶段：顺序执行与竞争 (Execution Phase) ---
        # 此时尝试将所有决策应用到环境中，排在后面的任务可能会因为
        # 资源在前几毫秒被前面的任务抢占而失败 (模拟并发冲突)
        for task, old_state, action_node in batch_decisions:
            total_attempts += 1

            # 执行动作 (此时环境资源才真正发生变化)
            next_state, reward, done, info = env.step(action_node, task)

            # 这里的 reward 可能会很低，因为如果发生了并发冲突，
            # env.step 会返回 "CPU Full" 或 "Bandwidth Unavailable" 的惩罚

            # 存储经验
            agent.remember(old_state, action_node, reward, next_state, done)
            loss = agent.replay()

            # 记录数据
            step_rewards.append(reward)
            if loss is not None:
                step_losses.append(loss)

            if info['status'] == 'Success':
                success_count += 1
                step_costs.append(info['cost'])
                active_tasks.append({
                    'end_time': time_step + task['duration'],
                    'node_id': action_node,
                    'cpu': task['cpu'],
                    'bw': task['bw'],
                    'path': info['path']
                })
            else:
                # 可选：打印冲突日志，观察并发竞争情况
                print(f"并发冲突: 节点 {action_node} 被抢占，任务失败: {info['status']}")
                pass

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