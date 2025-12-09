import numpy as np
import random
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent
import tensorflow as tf  # 引入 TF 用于记录
import shutil
import os

# MODEL_PATH = "dqn_trained_model.h5"
RESUME_TRAINING = False  # 如果要恢复训练，改为 True
CHECKPOINT_PATH = "logs/DQN_2025xxxx-xxxx/checkpoint_5000.h5" # 指向你要恢复的那个文件

def train():
    env = NetworkEnvironment()
    agent = DQNAgent()

    # --- 【新增】断点续训逻辑 ---
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"=== 正在从断点恢复: {CHECKPOINT_PATH} ===")
        agent.load(CHECKPOINT_PATH)

        # 【关键】恢复 Epsilon！
        # 如果你不知道具体断掉时是多少，可以给一个较低的值
        # 或者根据文件名里的步数估算：例如 5000 步
        # 假设之前的 decay 是 0.999，5000步后大约是 0.01 (举例)
        # 这里建议直接手动设置一个较低的值，比如 0.1 或 0.01
        agent.epsilon = 0.1
        print(f"=== Epsilon 已重置为: {agent.epsilon} (避免破坏已有模型) ===")
    else:
        print("=== 开始新的训练 (从零开始) ===")

    active_tasks = []

    # 统计变量
    step_rewards = []
    step_losses = []
    total_attempts = 0
    success_count = 0
    step_costs = []

    print(f"=== 开始训练 (总步数: {config.MAX_STEPS}) ===")
    print(f"=== TensorBoard 日志目录: {agent.log_dir} ===")

    for time_step in range(config.MAX_STEPS):

        # 1. 资源释放
        for i in range(len(active_tasks) - 1, -1, -1):
            task = active_tasks[i]
            if task['end_time'] <= time_step:
                env.release_task_resources(task)
                active_tasks.pop(i)

        # 2. 任务生成与训练
        loss = None

        if random.random() < config.TASK_GENERATION_PROB:
            total_attempts += 1
            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            state = env.get_global_state()
            action_node = agent.act(state)

            next_state, reward, done, info = env.step(action_node, new_task)

            agent.remember(state, action_node, reward, next_state, done)

            # 获取训练 Loss
            loss = agent.replay()

            # 记录基础指标
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

        if time_step % 10000 == 0 and time_step > 0:  # 每1000步保存一次
            # 保存临时模型
            ckpt_path = os.path.join(agent.log_dir, f"checkpoint_{time_step}.h5")
            agent.save(ckpt_path)
            print(f"【自动存档】已保存检查点: {ckpt_path}")

        # 3. 【TensorBoard】定期写入日志 (每100步)
        if time_step % 100 == 0 and time_step > 0:
            agent.update_target_model()

            # 计算平均值
            avg_reward = np.mean(step_rewards) if step_rewards else 0
            avg_loss = np.mean(step_losses) if step_losses else 0
            avg_cost = np.mean(step_costs) if step_costs else 0
            current_success_rate = success_count / total_attempts if total_attempts > 0 else 0

            # 使用 writer 写入
            with agent.summary_writer.as_default():
                tf.summary.scalar('Main/Average_Reward', avg_reward, step=time_step)
                tf.summary.scalar('Main/Loss', avg_loss, step=time_step)
                tf.summary.scalar('Performance/Success_Rate', current_success_rate, step=time_step)
                tf.summary.scalar('Performance/Average_Cost', avg_cost, step=time_step)
                tf.summary.scalar('Parameters/Epsilon', agent.epsilon, step=time_step)

            print(
                f"Step {time_step} | Reward: {avg_reward:.2f} | Loss: {avg_loss:.4f} | Success: {current_success_rate:.2%}")

            # 重置部分统计列表以观察局部趋势 (可选)
            step_rewards = []
            step_losses = []
            step_costs = []

    print("-" * 50)
    # agent.save(MODEL_PATH)
    # print("训练完成，模型已保存。")
    # === 【修改点】动态构建保存路径 ===
    # 1. 定义文件名
    model_filename = "dqn_trained_model.h5"

    # 2. 拼接路径: logs/DQN_时间戳/dqn_trained_model.h5
    # 注意：agent.log_dir 就是你之前在 dqn_agent.py 里定义的那个目录
    save_path = os.path.join(agent.log_dir, model_filename)

    # 3. 保存模型到该路径
    agent.save(save_path)

    print(f"训练完成！")
    print(f"模型已保存至目录: {agent.log_dir}")
    print(f"完整路径: {save_path}")
    # 把当前的 config 也备份进去，方便以后查参数
    shutil.copy('config.py', agent.log_dir)

if __name__ == "__main__":
    train()