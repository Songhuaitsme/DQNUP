import networkx as nx
import config
from data_loader import DataLoader

class TopologyManager:
    def __init__(self):
        self.data = DataLoader.load_network_topology()
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """构建图并初始化带宽和费用状态"""
        self.graph.add_nodes_from(self.data["vertices"])
        for u, v, dist, cap in self.data["edges"]:
            '''# 边属性:
            # weight = 物理距离 (用于计算延迟，或传统的SPF)
            # cost   = 逻辑费用 (用于最小费用寻路，初始可设为等于距离或固定值)
            # capacity = 总带宽
            # flow = 当前已用带宽'''
            # 策略：默认 Cost = Distance (你可以改为 1.0 来寻找最小跳数，或自定义价格)
            initial_cost = dist

            self.graph.add_edge(u, v, weight=dist, cost=initial_cost, capacity=cap, flow=0.0)

    def get_available_bandwidth(self, u, v) -> float:
        """获取链路可用带宽"""
        edge = self.graph[u][v]
        return edge['capacity'] - edge['flow']

    def update_link_costs(self, usage_sensitive=False):
        """
        [新功能] 更新链路费用
        usage_sensitive: 如果为True，则根据链路拥堵程度提高费用 (拥塞定价)
        """
        for u, v in self.graph.edges():
            edge = self.graph[u][v]
            base_cost = edge['weight']  # 基础费用来源于物理距离

            if usage_sensitive:
                # 示例公式：费用随着负载指数级上升 (由拥堵导致的"软性"费用)
                usage_ratio = edge['flow'] / edge['capacity']
                # 加上一个拥堵惩罚因子
                new_cost = base_cost * (1 + 5.0 * (usage_ratio ** 2))
                edge['cost'] = new_cost
            else:
                # 静态费用
                edge['cost'] = base_cost

    def allocate_bandwidth(self, path: list, amount: float):
        """占用路径带宽"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['flow'] += amount

        # [可选] 分配带宽后，立即更新网络费用状态（如果启用了动态定价）
        #self.update_link_costs(usage_sensitive=True)

    def release_bandwidth(self, path: list, amount: float):
        """释放路径带宽"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['flow'] = max(0.0, self.graph[u][v]['flow'] - amount)

    def find_constrained_path(self, target_node: int, bandwidth_needed: float) -> list:
        """
        核心寻路逻辑 (MCPF - Minimum Cost Path First):
        1. 剪枝：仅保留剩余带宽 >= 需求带宽的边。
        2. 寻路：在剪枝后的图中寻找从节点0到目标的【Cost最小】路径。
        """
        # 1. 构建临时子图 (剪枝)
        valid_edges = []
        for u, v, data in self.graph.edges(data=True):
            available = data['capacity'] - data['flow']
            if available >= bandwidth_needed:
                valid_edges.append((u, v))

        temp_graph = self.graph.edge_subgraph(valid_edges)

        # 2. 最小费用路径搜索 (核心修改点)
        try:
            if not temp_graph.has_node(0) or not temp_graph.has_node(target_node):
                return None

            # 修改：weight 参数由 'weight' 改为 'cost'
            # 这意味着算法现在寻找 sum(cost) 最小的路径，而不是 sum(dist) 最小
            path = nx.shortest_path(temp_graph, source=0, target=target_node, weight='cost')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def calculate_path_distance(self, path: list) -> float:
        """计算路径的总物理距离 (保持不变，用于统计延迟)"""
        dist = 0
        for i in range(len(path) - 1):
            dist += self.graph[path[i]][path[i + 1]]['weight']
        return dist

    def calculate_path_cost(self, path: list) -> float:
        """[新增] 计算路径的总费用"""
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph[path[i]][path[i + 1]]['cost']
        return cost