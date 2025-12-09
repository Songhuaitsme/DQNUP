import pandas as pd
from typing import Dict, List, Tuple
import config

class DataLoader:
    @staticmethod
    def load_network_topology() -> Dict:
        """加载12节点拓扑 (保持不变)"""
        raw_edges = [
            (0, 1, 110), (0, 2, 100), (1, 3, 30),
            (1, 6, 20), (2, 3, 50), (2, 9, 10),
            (3, 9, 20), (3, 8, 10), (3, 4, 40),
            (4, 6, 15), (4, 5, 20), (4, 10, 10),
            (5, 9, 10), (5, 11, 20), (6, 7, 20)
        ]
        edges_with_capacity = []
        for u, v, dist in raw_edges:
            edges_with_capacity.append(
                (u, v, dist, config.DEFAULT_LINK_BANDWIDTH)
                #带宽是默认带宽==能否进行定制化
            )
        return {
            "vertices": list(range(config.NODE_NUM)),
            "edges": edges_with_capacity
        }

    @staticmethod
    def load_base_prices() -> Dict[int, Dict[str, float]]:
        """
        加载表3中的资源基础价格 返回结构: {node_id: {'cpu': 0.1164, 'bw': 0.334}}
        """
        # 0号调度器价格为0
        prices = {0: {'cpu': 0.0, 'bw': 0.0}}

        # 表3 数据 (数据中心 1-11)
        # 格式: ID: (CPU价格, 带宽价格)
        table_data = {
            1: (0.1164, 0.334),
            2: (0.1122, 0.456),
            3: (0.1063, 0.385),
            4: (0.1102, 0.489),
            5: (0.1190, 0.399),
            6: (0.1010, 0.367),
            7: (0.1250, 0.347),
            8: (0.1050, 0.397),
            9: (0.1110, 0.343),
            10: (0.1220, 0.385),
            11: (0.1170, 0.356)
        }

        for node_id, (cpu_p, bw_p) in table_data.items():
            prices[node_id] = {
                'cpu': cpu_p,
                'bw': bw_p
            }

        return prices