# network_env.py
import numpy as np
import math
import config
from data_loader import DataLoader
from topology_manager import TopologyManager


class NetworkEnvironment:
    def __init__(self):
        self.topo_manager = TopologyManager()
        # 加载基础价格字典
        self.node_specs = DataLoader.load_base_prices()

        # 初始化节点资源状态
        self.node_resources = {}
        for i in range(config.NODE_NUM):
            self.node_resources[i] = {
                'total': config.DEFAULT_NODE_CPU,
                'used': 0.0
            }

    def get_dynamic_cpu_price(self, node_id: int) -> float:
        """
        计算动态 CPU 价格
        公式: P_final = P_base * (1 + alpha * usage_ratio^beta)
        """
        if node_id == 0: return 0.0

        res = self.node_resources[node_id]
        usage_ratio = min(1.0, res['used'] / res['total'])
        base_cpu_price = self.node_specs[node_id]['cpu']

        dynamic_p = base_cpu_price * (1 + config.PRICE_ALPHA * math.pow(usage_ratio, config.PRICE_BETA))
        return dynamic_p

    def get_global_state(self, current_loc: int = 0, task: dict = None) -> np.ndarray:
        """
        构建状态向量 State = [Location(One-hot), Node_Load_Rates, Task_Features]
        """
        # 1. 位置向量 (One-hot)
        loc_vec = np.zeros(config.NODE_NUM)
        loc_vec[current_loc] = 1.0

        # 2. 节点负载向量
        usage_vec = []
        for i in range(config.NODE_NUM):
            u = self.node_resources[i]['used'] / self.node_resources[i]['total']
            usage_vec.append(u)

        # 3. 任务特征 (归一化处理)
        if task is None:
            task_vec = [0.0, 0.0]
        else:
            # 简单的归一化: 假设最大 CPU 20, 最大带宽 50
            norm_cpu = task['cpu'] / 20.0
            norm_bw = task['bw'] / 50.0
            task_vec = [norm_cpu, norm_bw]

        return np.concatenate([loc_vec, np.array(usage_vec), np.array(task_vec)])

    def step(self, action_node_id: int, task: dict):
        """
        环境交互核心步骤
        Action: 目标节点 ID
        """
        # --- 0. 合法性检查 ---
        if action_node_id == 0:
            return self.get_global_state(), -10, True, {"status": "Invalid Target 0"}

        # --- 1. 资源检查 (CPU) ---
        node_res = self.node_resources[action_node_id]
        if (node_res['total'] - node_res['used']) < task['cpu']:
            return self.get_global_state(), -5, True, {"status": "CPU Full"}

        # --- 2. 路由检查 (Bandwidth) ---
        path = self.topo_manager.find_constrained_path(action_node_id, task['bw'])
        if path is None:
            return self.get_global_state(), -5, True, {"status": "Bandwidth/Path Unavailable"}

        # --- 3. 资源分配 ---
        self.node_resources[action_node_id]['used'] += task['cpu']
        self.topo_manager.allocate_bandwidth(path, task['bw'])

        # --- 4. 计算指标与奖励 ---
        # 负载均衡分 (越空闲分数越高 0~1)
        current_usage = self.node_resources[action_node_id]['used'] / self.node_resources[action_node_id]['total']
        load_balance_score = 1.0 - current_usage

        # 成本计算 (简化版，包含动态CPU价格和固定带宽价格)
        unit_cpu_price = self.get_dynamic_cpu_price(action_node_id)
        raw_cost = (unit_cpu_price * task['cpu'] + self.node_specs[action_node_id]['bw'] * task['bw'])
        cost_score = raw_cost * 0.1  # 缩放系数

        # 最终奖励函数: 基础奖励 + 负载均衡奖励 - 成本惩罚
        reward = 10.0 + (2.0 * load_balance_score) - cost_score

        info = {
            "status": "Success",
            "path": path,
            "cost": cost_score * 10.0,  # 还原显示用的 Cost
            "unit_price": unit_cpu_price
        }

        # 任务分配后，状态会更新，但对于单步决策任务，Done=True
        return self.get_global_state(action_node_id, None), reward, True, info

    def release_task_resources(self, task_info):
        """释放过期任务占用的 CPU 和 带宽"""
        node_id = task_info['node_id']
        cpu = task_info['cpu']
        path = task_info['path']
        bw = task_info['bw']

        self.node_resources[node_id]['used'] = max(0, self.node_resources[node_id]['used'] - cpu)
        self.topo_manager.release_bandwidth(path, bw)