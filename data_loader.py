# data_loader.py
import config
from typing import Dict, List, Tuple


class DataLoader:

    @staticmethod
    def load_node_capacities() -> Dict[int, float]:
        """
        加载每个节点的差异化 CPU 总量
        返回格式: {node_id: total_cpu}
        """
        # 初始化所有节点为默认值
        capacities = {i: config.DEFAULT_NODE_CPU for i in range(config.NODE_NUM)}

        # 自定义节点规格 (模拟异构集群)
        custom_specs = {
            0: 0,  # 调度器本身不计算
            1: 120, 2: 120, 3: 60, 4: 60,
            5: 240, 6: 80, 7: 130, 8: 180,
            9: 70, 10: 110, 11: 60
        }

        capacities.update(custom_specs)
        return capacities

    @staticmethod
    def load_network_topology() -> Dict:
        """
        加载12节点拓扑结构
        返回: {'vertices': List, 'edges': List[(u, v, dist, cap)]}
        """
        # (起点, 终点, 距离/延迟, 总带宽)
        raw_edges = [
            (0, 1, 110, 2000), (0, 2, 100, 2000), (1, 3, 30, 500),
            (1, 6, 20, 400), (2, 3, 50, 600), (2, 9, 10, 300),
            (3, 9, 20, 400), (3, 8, 10, 500), (3, 4, 40, 420),
            (4, 6, 15, 380), (4, 5, 20, 320), (4, 10, 10, 450),
            (5, 9, 10, 540), (5, 11, 20, 490), (6, 7, 20, 500)
        ]
        return {
            "vertices": list(range(config.NODE_NUM)),
            "edges": raw_edges
        }

    @staticmethod
    def load_base_prices() -> Dict[int, Dict[str, float]]:
        """
        加载资源基础价格
        返回结构: {node_id: {'cpu': float, 'bw': float}}
        """
        prices = {0: {'cpu': 0.0, 'bw': 0.0}}

        # 数据中心 1-11 的基础价格 (CPU价格, 带宽价格)
        table_data = {
            1: (0.1164, 0.334), 2: (0.1122, 0.456), 3: (0.1063, 0.385),
            4: (0.1102, 0.489), 5: (0.1190, 0.399), 6: (0.1010, 0.367),
            7: (0.1250, 0.347), 8: (0.1050, 0.397), 9: (0.1110, 0.343),
            10: (0.1220, 0.385), 11: (0.1170, 0.356)
        }

        for node_id, (cpu_p, bw_p) in table_data.items():
            prices[node_id] = {'cpu': cpu_p, 'bw': bw_p}

        return prices