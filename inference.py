import numpy as np
import random
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent

# 指定要加载的模型
MODEL_PATH = "dqn_trained_model.h5"
# 推理测试的步数
TEST_STEPS = 1000


def inference():
    print(f"=== 开始推理测试 (加载模型: {MODEL_PATH}) ===")

    env = NetworkEnvironment()
    agent = DQNAgent()

    # 【关键】加载训练好的模型
    try:
        agent.load(MODEL_PATH)
    except Exception as e:
        print(f"加载失败，请先运行 train.py: {e}")
        return

    # 【关键】强制关闭随机探索，只选模型认为最好的
    agent.epsilon = 0.0

    active_tasks = []
    success_count = 0
    fail_count = 0
    total_cost = 0

    for time_step in range(TEST_STEPS):

        # 1. 资源释放
        for i in range(len(active_tasks) - 1, -1, -1):
            task = active_tasks[i]
            if task['end_time'] <= time_step:
                env.release_task_resources(task)
                active_tasks.pop(i)

        # 2. 任务生成与决策
        if random.random() < config.TASK_GENERATION_PROB:
            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            state = env.get_global_state()

            # 这里 act 完全基于模型预测，没有随机性
            action_node = agent.act(state)

            next_state, reward, done, info = env.step(action_node, new_task)

            # 【注意】推理阶段不需要 agent.remember() 和 agent.replay()

            if info['status'] == 'Success':
                success_count += 1
                total_cost += info['cost']
                active_tasks.append({
                    'end_time': time_step + new_task['duration'],
                    'node_id': action_node,
                    'cpu': new_task['cpu'],
                    'bw': new_task['bw'],
                    'path': info['path']
                })
                print(f"Step {time_step}: [成功] 节点 {action_node}, 路径 {info['path']}, 成本 {info['cost']:.2f}")
            else:
                fail_count += 1
                print(f"Step {time_step}: [失败] 原因: {info['status']}")

    print("-" * 50)
    print("=== 推理测试报告 ===")
    total_tasks = success_count + fail_count
    print(f"总请求任务: {total_tasks}")
    print(f"成功分配: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"成功率: {success_count / (total_tasks + 1e-5):.2%}")
    if success_count > 0:
        print(f"平均任务成本: {total_cost / success_count:.2f}")


if __name__ == "__main__":
    inference()