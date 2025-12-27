# inference.py
import numpy as np
import random
import matplotlib.pyplot as plt
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent

# ================= 推理配置 =================
MODEL_PATH = "dqn_trained_model.h5"  # 指定模型文件
TEST_STEPS = 1000  # 测试步数


def inference():
    print(f"=== 开始推理测试 (加载模型: {MODEL_PATH}) ===")

    env = NetworkEnvironment()
    agent = DQNAgent()

    try:
        agent.load(MODEL_PATH)
    except Exception as e:
        print(f"加载失败，请检查文件路径或先运行 train.py。错误信息: {e}")
        return

    # 【关键】强制关闭随机探索，完全依赖模型决策
    agent.epsilon = 0.0

    active_tasks = []
    success_count = 0
    fail_count = 0
    total_cost = 0

    # 记录节点利用率用于绘图
    node_usage_history = {i: [] for i in range(config.NODE_NUM)}

    for time_step in range(TEST_STEPS):

        # 1. 资源释放
        for i in range(len(active_tasks) - 1, -1, -1):
            task = active_tasks[i]
            if task['end_time'] <= time_step:
                env.release_task_resources(task)
                active_tasks.pop(i)

        # 2. 任务生成 (保持与训练一致的泊松分布，或自定义压力测试)
        current_batch_size = np.random.poisson(config.TASK_ARRIVAL_RATE)

        # 记录当前时刻各节点利用率
        for i in range(config.NODE_NUM):
            res = env.node_resources[i]
            usage = res['used'] / res['total']
            node_usage_history[i].append(usage)

        # 处理任务
        for i in range(current_batch_size):
            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            state = env.get_global_state(0, task=new_task)
            action_node = agent.act(state)
            next_state, reward, done, info = env.step(action_node, new_task)

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
                print(f"Step {time_step}-{i}: [成功] 节点 {action_node}, 成本 {info['cost']:.2f}")
            else:
                fail_count += 1
                print(f"Step {time_step}-{i}: [失败] {info['status']}")

    # 3. 输出报告
    print("-" * 50)
    print("=== 推理测试报告 ===")
    total_tasks = success_count + fail_count
    print(f"总请求任务: {total_tasks}")
    print(f"成功分配: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"成功率: {success_count / (total_tasks + 1e-5):.2%}")
    if success_count > 0:
        print(f"平均任务成本: {total_cost / success_count:.2f}")

    # 4. 绘制利用率曲线
    print("正在生成节点利用率曲线图...")
    plt.figure(figsize=(12, 6))

    for node_id, usages in node_usage_history.items():
        # 可选：跳过 0 号调度器节点
        if node_id == 0: continue
        plt.plot(usages, label=f'Node {node_id}')

    plt.title(f'Node CPU Utilization Over Time (Steps={TEST_STEPS})')
    plt.xlabel('Time Step')
    plt.ylabel('CPU Utilization Ratio')
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1))
    plt.tight_layout()

    plt.savefig('node_utilization.png', dpi=300)
    plt.show()
    print("图表已保存为 node_utilization.png")


if __name__ == "__main__":
    inference()