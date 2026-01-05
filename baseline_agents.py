# baseline_agents.py
import numpy as np
import config
import hashlib

class BaselineAgent:
    def __init__(self, env):
        """
        初始化基准算法代理
        :param env: NetworkEnvironment 实例，用于读取拓扑和资源状态
        """
        self.env = env
        # 候选节点是 1 到 11 (0是调度器，不可作为目标)
        self.candidates = list(range(1, config.NODE_NUM))

    def act_spf(self, task: dict) -> int:
        """
        SPF (Shortest Path First) 策略:
        1. 筛选出 CPU 和 带宽都满足任务要求的节点。
        2. 在可行节点中，选择距离调度器(节点0)物理路径最短的节点。
        """
        best_node = -1
        min_dist = float('inf')

        for node_id in self.candidates:
            # 1. 检查 CPU 资源是否足够
            if not self._check_cpu_available(node_id, task['cpu']):
                continue

            # 2. 检查带宽并获取路径 (复用环境的寻路逻辑)
            path = self.env.topo_manager.find_constrained_path(node_id, task['bw'])
            if path is None:
                continue

            # 3. 计算路径物理距离 (Weight)
            dist = self.env.topo_manager.calculate_path_distance(path)

            # 4. 择优: 距离更短
            if dist < min_dist:
                min_dist = dist
                best_node = node_id

        # 如果没有可行节点，随机返回一个（环境会处理失败）
        if best_node == -1:
            # 方案 A: 仍然随机，但引入算法特有的噪声，避免随机数序列撞车
            # return np.random.choice(self.candidates)

            # 方案 B (更推荐): 既然找不到合适节点，就随机选一个，但打印日志
            # print("SPF 找不到可行解，随机盲选...")
            return np.random.choice(self.candidates)

        return best_node

    def act_crf(self, task: dict) -> int:
        """
        CRF (Compute Resource First) 策略:
        1. 筛选出 CPU 和 带宽都满足任务要求的节点。
        2. 在可行节点中，选择当前 CPU 资源价格最低的节点。
           (注: 你的环境设定中，价格与负载挂钩，因此这也隐含了负载均衡)
        """
        best_node = -1
        min_price = float('inf')

        for node_id in self.candidates:
            # 1. 检查 CPU
            if not self._check_cpu_available(node_id, task['cpu']):
                continue

            # 2. 检查带宽 (必须保证路径可达)
            path = self.env.topo_manager.find_constrained_path(node_id, task['bw'])
            if path is None:
                continue

            # 3. 获取当前节点的动态资源价格
            price = self.env.get_dynamic_cpu_price(node_id)

            # 4. 择优: 价格更低
            if price < min_price:
                min_price = price
                best_node = node_id

        if best_node == -1:
            return np.random.choice(self.candidates)

        return best_node

    def _check_cpu_available(self, node_id, cpu_demand):
        """辅助函数: 检查节点剩余 CPU"""
        res = self.env.node_resources[node_id]
        return (res['total'] - res['used']) >= cpu_demand