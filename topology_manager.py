import networkx as nx
import config
from data_loader import DataLoader

class TopologyManager:
    def __init__(self):
        self.data = DataLoader.load_network_topology()
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """构建图，初始化带宽状态"""
        self.graph.add_nodes_from(self.data["vertices"])
        for u, v, dist, cap in self.data["edges"]:
            # 边属性：weight=距离, capacity=总带宽, flow=当前占用
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
            # 保护性代码，防止减成负数
            self.graph[u][v]['flow'] = max(0.0, self.graph[u][v]['flow'] - amount)

    def find_constrained_path(self, target_node: int, bandwidth_needed: float) -> list:
        """
        核心寻路逻辑：
        1. 寻找从节点0到 target_node 的路径
        2. 约束：路径上所有边的可用带宽 >= bandwidth_needed
        3. 优化目标：路径总距离最短
        """
        # 1. 剪枝：构建临时图，只包含带宽充足的边
        valid_edges = []
        for u, v, data in self.graph.edges(data=True):
            available = data['capacity'] - data['flow']
            if available >= bandwidth_needed:
                valid_edges.append((u, v))

        temp_graph = self.graph.edge_subgraph(valid_edges)

        # 2. 在剪枝后的图中找最短路
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