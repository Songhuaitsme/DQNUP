'''#节点分配结果可视化
1.全节点负载耦合图示
2.各节点热力图耦合
3.各节点负载图
4.所有节点平均负载
5.切片利用率'''

import numpy as np
import random
import matplotlib.pyplot as plt
import config
from network_env import NetworkEnvironment
from dqn_agent import DQNAgent
import seaborn as sns

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
    '''print("正在生成节点利用率曲线图...")
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
'''

    # -------------------------------------------------------
    # 改进方案 1: 节点负载热力图 (Heatmap)
    # -------------------------------------------------------
    '''# print("正在生成节点利用率热力图...")
    # import seaborn as sns  # 如果没有安装 seaborn，请 pip install seaborn
    # 
    # # 1. 数据准备：转换为 (节点数 x 时间步) 的矩阵
    # # 过滤掉 Node 0 (调度器)，只看计算节点 1-11
    # target_nodes = sorted([n for n in node_usage_history.keys() if n != 0])
    # data_matrix = []
    # 
    # for node_id in target_nodes:
    #     data_matrix.append(node_usage_history[node_id])
    # 
    # data_matrix = np.array(data_matrix)  # 形状: (11, 1000)
    # 
    # # 2. 绘图
    # plt.figure(figsize=(15, 6))
    # ax = sns.heatmap(data_matrix,
    #                  cmap="RdYlBu_r",  # 红黄蓝反转：红色代表高负载，蓝色代表低负载
    #                  vmin=0, vmax=1,  # 固定范围 0~1
    #                  yticklabels=[f"Node {n}" for n in target_nodes],
    #                  cbar_kws={'label': 'CPU Utilization'})
    # 
    # plt.title("Node CPU Utilization Heatmap (Red=High Load)")
    # plt.xlabel("Time Step")
    # plt.ylabel("Node ID")
    # plt.tight_layout()
    # plt.savefig('node_utilization_heatmap.png', dpi=300)
    # plt.show()
    # print("图表已保存为 node_utilization_heatmap.png")
'''

    # -------------------------------------------------------
    # 改进方案 2: 分离多子图 (Subplots)
    # -------------------------------------------------------
    print("正在生成节点独立利用率图...")

    target_nodes = sorted([n for n in node_usage_history.keys() if n != 0])
    num_nodes = len(target_nodes)

    # 创建 4行 x 3列 的网格
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # 展平方便遍历

    for idx, node_id in enumerate(target_nodes):
        ax = axes[idx]
        ax.plot(node_usage_history[node_id], color='tab:blue', linewidth=1)
        ax.set_title(f"Node {node_id}", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # 添加一条参考线，例如 0.8 是高负载警戒线
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, linewidth=0.8)

    # 隐藏多余的空子图
    for i in range(num_nodes, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"Individual Node CPU Utilization (Steps={TEST_STEPS})", fontsize=14)
    plt.tight_layout()
    plt.savefig('node_utilization_subplots.png', dpi=300)
    plt.show()
    print("图表已保存为 node_utilization_subplots.png")


    # -------------------------------------------------------
    # 改进方案 3: 集群平均负载与范围 (Mean + Min/Max)
    # -------------------------------------------------------
    print("正在生成集群聚合统计图...")

    target_nodes = sorted([n for n in node_usage_history.keys() if n != 0])
    # 转换为矩阵方便计算
    all_data = np.array([node_usage_history[n] for n in target_nodes])  # Shape: (11, 1000)

    # 计算指标
    avg_usage = np.mean(all_data, axis=0)  # 每个时刻的平均值
    max_usage = np.max(all_data, axis=0)  # 每个时刻的最大值
    min_usage = np.min(all_data, axis=0)  # 每个时刻的最小值

    plt.figure(figsize=(12, 6))

    # 1. 绘制范围 (最大值和最小值之间的区域) - 体现负载均衡度，区域越宽说明越不均衡
    plt.fill_between(range(TEST_STEPS), min_usage, max_usage,
                     color='gray', alpha=0.2, label='Usage Range (Min-Max)')

    # 2. 绘制平均线
    plt.plot(avg_usage, color='blue', linewidth=2, label='Cluster Average')

    # 3. (可选) 绘制最忙碌的节点作为参考
    # plt.plot(max_usage, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Max Node Load')

    plt.title("Cluster CPU Utilization Summary")
    plt.xlabel("Time Step")
    plt.ylabel("Utilization Ratio")
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('node_utilization_summary.png', dpi=300)
    plt.show()
    print("图表已保存为 node_utilization_summary.png")


    # -------------------------------------------------------
    #改进方案: 局部放大折线图 (Zoom-in Snapshot)
    # -------------------------------------------------------
    '''print("正在生成局部时间步折线图...")

    # === 配置区域 ===
    START_STEP = 450 # 开始步数 (您可以改为 100, 500 等)
    WINDOW_SIZE = 50 # 展示的时间步长度
    END_STEP = min(START_STEP + WINDOW_SIZE, TEST_STEPS)
    # ===============

    plt.figure(figsize=(14, 7))

    # 准备不同形状的标记点，方便区分不同线条
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']

    for node_id, usages in node_usage_history.items():
        # 跳过调度器节点0
        if node_id == 0: continue

        # 【关键步骤】只截取 slice_data
        slice_data = usages[START_STEP: END_STEP]
        x_axis = range(START_STEP, END_STEP)

        plt.plot(x_axis, slice_data,
                 label=f'Node {node_id}',
                 marker=markers[node_id % len(markers)],  # 添加点标记
                 markersize=6,  # 标记大小
                 linewidth=2,  # 线宽
                 alpha=0.8)  # 透明度

    plt.title(f'Node CPU Utilization (Step {START_STEP} - {END_STEP})', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('CPU Utilization Ratio', fontsize=12)
    plt.ylim(-0.05, 1.1)  # 稍微留一点余地

    # 设置X轴刻度为整数
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()

    plt.savefig('node_utilization_snapshot.png', dpi=300)
    plt.show()
    print(f"图表已保存为 node_utilization_snapshot.png (展示范围: {START_STEP}-{END_STEP})")
'''


if __name__ == "__main__":
    inference()