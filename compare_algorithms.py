# compare_algorithms.py
import numpy as np
import random
import matplotlib.pyplot as plt
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent
from baseline_agents import BaselineAgent  # 导入刚刚创建的基准类

# ================= 测试配置 =================
MODEL_PATH = "dqn_trained_model.h5"
TEST_STEPS = 1000  # 测试步数
RANDOM_SEED = 42  # 固定种子，保证任务序列一致


def run_simulation(algo_name, agent, seed):
    """
    运行单个算法的仿真测试
    """
    print(f"\n=== 正在运行算法: {algo_name} ===")

    # 1. 重置环境和随机种子 (关键! 确保每个算法面对相同的任务流)
    env = NetworkEnvironment()
    np.random.seed(seed)
    random.seed(seed)

    active_tasks = []
    success_count = 0
    fail_count = 0
    total_cost = 0.0
    total_reward = 0.0

    # 用于绘图的数据记录
    cost_history = []
    success_rate_history = []

    for time_step in range(TEST_STEPS):
        # --- 资源释放 ---
        for i in range(len(active_tasks) - 1, -1, -1):
            task = active_tasks[i]
            if task['end_time'] <= time_step:
                env.release_task_resources(task)
                active_tasks.pop(i)

        # --- 任务生成 (泊松分布) ---
        current_batch_size = np.random.poisson(config.TASK_ARRIVAL_RATE)

        for i in range(current_batch_size):
            new_task = {
                'cpu': random.randint(*config.TASK_CPU_DEMAND),
                'bw': random.randint(*config.TASK_BW_DEMAND),
                'duration': int(np.random.exponential(config.TASK_DURATION_MEAN)) + 1
            }

            # --- 决策 ---
            if algo_name == "DQN":
                state = env.get_global_state(0, task=new_task)
                action_node = agent.act(state)  # DQN 预测
            elif algo_name == "SPF":
                action_node = agent.act_spf(new_task)  # 基准 SPF
            elif algo_name == "CRF":
                action_node = agent.act_crf(new_task)  # 基准 CRF

            # --- 执行 ---
            next_state, reward, done, info = env.step(action_node, new_task)

            # --- 统计 ---
            total_reward += reward
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
            else:
                fail_count += 1

        # 记录每一步的累计平均指标
        current_total = success_count + fail_count
        if current_total > 0:
            cost_history.append(total_cost / current_total)  # 平均成本
            success_rate_history.append(success_count / current_total)  # 成功率
        else:
            cost_history.append(0)
            success_rate_history.append(1.0)

    # 汇总结果
    total_requests = success_count + fail_count
    avg_cost = total_cost / success_count if success_count > 0 else 0
    success_rate = success_count / total_requests if total_requests > 0 else 0

    print(f"[{algo_name}] 完成。成功率: {success_rate:.2%}, 平均成本: {avg_cost:.2f}, 总奖励: {total_reward:.2f}")

    return {
        "success_rate": success_rate,
        "avg_cost": avg_cost,
        "cost_history": cost_history,
        "success_rate_history": success_rate_history
    }


def main():
    # 1. 初始化智能体
    # --- DQN ---
    dqn_agent = DQNAgent()
    try:
        dqn_agent.load(MODEL_PATH)
        dqn_agent.epsilon = 0.0  # 强制关闭探索
    except Exception as e:
        print(f"无法加载DQN模型: {e}")
        return

    # --- 基准算法代理 ---
    # 这里我们传入一个临时的 env 实例给 Agent 初始化用，实际运行时 run_simulation 会创建新的 env
    temp_env = NetworkEnvironment()
    baseline_agent = BaselineAgent(temp_env)

    # 2. 运行对比测试
    results = {}

    # 运行 DQN
    results['DQN'] = run_simulation("DQN", dqn_agent, RANDOM_SEED)

    # 运行 SPF (需要将 baseline_agent 传入，注意 SPF 使用 baseline_agent.act_spf)
    # 为了复用逻辑，我们在 run_simulation 内部做了 getattr 判断，或者直接传对象
    results['SPF'] = run_simulation("SPF", baseline_agent, RANDOM_SEED)

    # 运行 CRF
    results['CRF'] = run_simulation("CRF", baseline_agent, RANDOM_SEED+1)

    # 3. 绘图对比
    print("\n正在生成对比图表...")

    # 图1: 平均成本随时间变化
    plt.figure(figsize=(12, 5))
    for name, res in results.items():
        plt.plot(res['cost_history'], label=name, linewidth=2)
    plt.title(f'Average Cost Comparison (Lower is Better)')
    plt.xlabel('Time Step')
    plt.ylabel('Avg Cost per Task')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison_cost.png', dpi=300)

    # 图2: 成功率随时间变化
    plt.figure(figsize=(12, 5))
    for name, res in results.items():
        plt.plot(res['success_rate_history'], label=name, linewidth=2)
    plt.title(f'Success Rate Comparison (Higher is Better)')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison_success_rate.png', dpi=300)

    print("图表已保存: comparison_cost.png, comparison_success_rate.png")


if __name__ == "__main__":
    main()