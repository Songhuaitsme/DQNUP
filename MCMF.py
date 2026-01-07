""""最小费用最大流"""
import networkx as nx
from data_loader import DataLoader
import pandas as pd


def calculate_mcmf():
    # 1. 加载数据
    topo_data = DataLoader.load_network_topology()

    # 2. 构建有向图 (DiGraph) 用于流计算
    # NetworkX 的流算法需要有向图。由于 topology_manager 使用无向图，
    # 我们将无向边拆解为两条容量相同的有向边。
    G = nx.DiGraph()

    # 添加节点
    G.add_nodes_from(topo_data['vertices'])

    # 添加边 (u, v, cost, capacity)
    for u, v, dist, cap in topo_data['edges']:
        # 正向边
        G.add_edge(u, v, weight=dist, capacity=cap)
        # 反向边 (假设全双工，容量相同；若是半双工则需共享容量，这里按标准网络流模型处理)
        G.add_edge(v, u, weight=dist, capacity=cap)

    source_node = 0
    results = []

    print(f"{'Target':<6} | {'Max Flow':<10} | {'Min Total Cost':<15} | {'Avg Cost/Unit':<15}")
    print("-" * 55)

    # 3. 对每个节点计算 MCMF
    for target_node in range(1, 12):  # 遍历 Node 1 到 11
        # 步骤 A: 计算最大流数值
        flow_value = nx.maximum_flow_value(G, source_node, target_node, capacity='capacity')

        # 步骤 B: 在强行要求达到最大流的情况下，计算最小费用
        # 这里的 min_cost_flow 可以在指定 flow 需求下找最小费
        # 我们构建一个需求字典：源点流出 flow_value，汇点流入 flow_value
        G_demand = G.copy()
        for node in G_demand.nodes():
            G_demand.nodes[node]['demand'] = 0

        G_demand.nodes[source_node]['demand'] = -flow_value  # 流出
        G_demand.nodes[target_node]['demand'] = flow_value  # 流入

        try:
            # 计算最小费用流
            min_cost = nx.min_cost_flow_cost(G_demand, demand='demand', weight='weight')

            # 记录详细路径信息 (可选，这里仅展示汇总)
            flow_dict = nx.min_cost_flow(G_demand, demand='demand', weight='weight')

            avg_cost = min_cost / flow_value if flow_value > 0 else 0

            results.append({
                "Target": target_node,
                "MaxFlow": flow_value,
                "MinCost": min_cost,
                "AvgCost": round(avg_cost, 2)
            })

            print(f"{target_node:<6} | {flow_value:<10} | {min_cost:<15} | {avg_cost:<15.2f}")

        except nx.NetworkXUnfeasible:
            print(f"{target_node:<6} | Unfeasible")

    return results


if __name__ == "__main__":
    calculate_mcmf()