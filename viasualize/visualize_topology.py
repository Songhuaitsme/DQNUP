import matplotlib.pyplot as plt
import networkx as nx


def visualize_optimized_topology():
    # 1. 定义数据 (来源于 data_loader.py)
    # 格式: (起点, 终点, 距离D, 带宽BW)
    raw_edges = [
        (0, 1, 110, 2000), (0, 2, 100, 2000), (1, 3, 30, 500),
        (1, 6, 20, 400), (2, 3, 50, 600), (2, 9, 10, 300),
        (3, 9, 20, 400), (3, 8, 10, 500), (3, 4, 40, 420),
        (4, 6, 15, 380), (4, 5, 20, 320), (4, 10, 10, 450),
        (5, 9, 10, 540), (5, 11, 20, 490), (6, 7, 20, 500)
    ]

    # 2. 构建图
    G = nx.Graph()
    for u, v, dist, cap in raw_edges:
        G.add_edge(u, v, weight=dist, capacity=cap)

    # 3. 布局设置 (关键点：均匀分布)
    # 使用 spring_layout，但增加 'k' 值（最佳距离），这会增加节点间的排斥力，使图看起来更舒展
    # seed=42 保证每次生成的图形状一样
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)

    # 4. 节点样式配置 (关键点：0节点特殊处理)
    # 列表推导式：如果节点是0，设为橙色，否则设为淡蓝
    node_colors = ['#ff9f43' if node == 0 else '#a2d1f0' for node in G.nodes()]
    # 列表推导式：如果节点是0，尺寸设为 1200，否则设为 800
    node_sizes = [1200 if node == 0 else 800 for node in G.nodes()]

    plt.figure(figsize=(14, 10))  # 画布大一点

    # 5. 绘制节点
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors='#2c3e50',  # 节点边框颜色
                           linewidths=2)  # 节点边框宽度

    # 6. 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', font_weight='bold')

    # 7. 绘制边
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

    # 8. 绘制边的标签 (D:距离, BW:带宽)
    edge_labels = {
        (u, v): f"D:{d}\nBW:{c}"
        for u, v, d, c in raw_edges
    }

    # bbox 参数给文字加一个白色背景框，防止挡住线条
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9,
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # 标题和收尾
    plt.title("Network Topology (Highlighed Entry Node 0)", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_optimized_topology()