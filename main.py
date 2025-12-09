# main.py
import numpy as np
import random
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent


def main():
    env = NetworkEnvironment()
    agent = DQNAgent()

    # 正在运行的任务列表: [{'end_time': 105, 'node_id': 3, 'cpu': 10, 'bw': 20, 'path': [...]}]
    active_tasks = []

    total_rewards = []
    success_count = 0
    fail_count = 0

    print(f"开始模拟，总步数: {config.MAX_STEPS}")
    print("-" * 50)

    for time_step in range(config.MAX_STEPS):

        # --- 1. 资源释放 (处理过期任务) ---
        # 倒序遍历以便安全删除
        for i in range(len(active_tasks) - 1, -1, -1):
            task = active_tasks[i]
            if task['end_time'] <= time_step:
                env.release_task_resources(task)
                active_tasks.pop(i)

        # --- 2. 任务生成 (概率生成) ---
        if random.random() < config.TASK_GENERATION_PROB:
            # 创建新任务
            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            # 获取当前状态
            state = env.get_global_state()

            # DQN 选择目标节点
            action_node = agent.act(state)

            # 环境执行
            next_state, reward, done, info = env.step(action_node, new_task)

            # 存储经验并学习
            agent.remember(state, action_node, reward, next_state, done)
            agent.replay()

            total_rewards.append(reward)

            if info['status'] == 'Success':
                success_count += 1
                # 记录任务以便后续释放
                active_tasks.append({
                    'end_time': time_step + new_task['duration'],
                    'node_id': action_node,
                    'cpu': new_task['cpu'],
                    'bw': new_task['bw'],
                    'path': info['path']
                })

                # 打印日志 (可选)
                if success_count % 50 == 0:
                    price = info['unit_price']
                    print(f"Step {time_step}: 成功分配到节点 {action_node}, 路径 {info['path']}, 动态单价 {price:.2f}")
            else:
                fail_count += 1
                # print(f"Step {time_step}: 分配失败 - {info}")

        # --- 3. 更新 Target Network ---
        if time_step % 100 == 0:
            agent.update_target_model()

    print("-" * 50)
    print("模拟结束")
    print(f"总任务数: {success_count + fail_count}")
    print(f"成功分配: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"成功率: {success_count / (success_count + fail_count + 1e-5):.2%}")


if __name__ == "__main__":
    main()