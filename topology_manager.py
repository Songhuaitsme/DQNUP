# topology_manager.py
import networkx as nx
import config
from data_loader import DataLoader

class TopologyManager:
    def __init__(self):
        self.data = DataLoader.load_network_topology()
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """构建图并初始化带宽状态"""
        self.graph.add_nodes_from(self.data["vertices"])
        for u, v, dist, cap in self.data["edges"]:
            # 边属性: weight=距离(用于最短路), capacity=总带宽, flow=当前已用带宽
            self.graph.add_edge(u, v, weight=dist, capacity=cap, flow=0.0)

    def get_available_bandwidth(self, u, v) -> float:
        """获取链路可用带宽"""
        edge = self.graph[u][v]
        return edge['capacity'] - edge['flow']

    def allocate_bandwidth(self, path: list, amount: float):
        """占用路径带宽"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            self.graph[u][v]['flow'] += amount

    def release_bandwidth(self, path: list, amount: float):
        """释放路径带宽"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # 保护性计算，防止浮点数误差导致负值
            self.graph[u][v]['flow'] = max(0.0, self.graph[u][v]['flow'] - amount)

    def find_constrained_path(self, target_node: int, bandwidth_needed: float) -> list:
        """
        核心寻路逻辑 (CSPF - Constrained Shortest Path First):
        1. 剪枝：仅保留剩余带宽 >= 需求带宽的边。
        2. 寻路：在剪枝后的图中寻找从节点0到目标的物理距离最短路径。
        """
        # 1. 构建临时子图 (剪枝)
        valid_edges = []
        for u, v, data in self.graph.edges(data=True):
            available = data['capacity'] - data['flow']
            if available >= bandwidth_needed:
                valid_edges.append((u, v))

        temp_graph = self.graph.edge_subgraph(valid_edges)

        # 2. 最短路径搜索
        try:
            if not temp_graph.has_node(0) or not temp_graph.has_node(target_node):
                return None
            path = nx.shortest_path(temp_graph, source=0, target=target_node, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def calculate_path_distance(self, path: list) -> float:
        """计算路径的总物理距离"""
        dist = 0
        for i in range(len(path) - 1):
            dist += self.graph[path[i]][path[i + 1]]['weight']
        return dist