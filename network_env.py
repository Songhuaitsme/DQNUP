import numpy as np
import math
import config
from data_loader import DataLoader
from topology_manager import TopologyManager

class NetworkEnvironment:
    def __init__(self):
        self.topo_manager = TopologyManager()
        # 加载新的价格字典 {'cpu': ..., 'bw': ...}
        self.node_specs = DataLoader.load_base_prices()

        self.node_resources = {}
        for i in range(config.NODE_NUM):
            self.node_resources[i] = {
                'total': config.DEFAULT_NODE_CPU,
                'used': 0.0
            }

    def get_dynamic_cpu_price(self, node_id: int) -> float:
        """
        仅计算 CPU 的动态价格 (带宽价格通常固定或另算)
        公式: P_cpu = Base_CPU * (1 + alpha * usage^beta)
        """
        if node_id == 0: return 0.0

        res = self.node_resources[node_id]
        usage_ratio = min(1.0, res['used'] / res['total'])

        # 获取该节点的 CPU 基础单价
        base_cpu_price = self.node_specs[node_id]['cpu']

        # 动态定价逻辑仅作用于 CPU
        dynamic_p = base_cpu_price * (1 + config.PRICE_ALPHA * math.pow(usage_ratio, config.PRICE_BETA))
        return dynamic_p

    def get_global_state(self, current_loc: int = 0) -> np.ndarray:
        # 保持不变
        loc_vec = np.zeros(config.NODE_NUM)
        loc_vec[current_loc] = 1.0
        usage_vec = []
        for i in range(config.NODE_NUM):
            u = self.node_resources[i]['used'] / self.node_resources[i]['total']
            usage_vec.append(u)
        return np.concatenate([loc_vec, np.array(usage_vec)])

    def step(self, action_node_id: int, task: dict):
        # 0. 基础检查 (保持不变)
        if action_node_id == 0:
            return self.get_global_state(), -1000, True, {"status": "Invalid Target 0"}

        # 1. 检查节点 CPU 资源 (保持不变)
        node_res = self.node_resources[action_node_id]
        if (node_res['total'] - node_res['used']) < task['cpu']:
            return self.get_global_state(), -2000, True, {"status": "CPU Full"}

        # 2. 寻找路径 (保持不变)
        path = self.topo_manager.find_constrained_path(action_node_id, task['bw'])
        if path is None:
            return self.get_global_state(), -2000, True, {"status": "Bandwidth/Path Unavailable"}

        # --- 分配成功，扣除资源 ---
        self.node_resources[action_node_id]['used'] += task['cpu']
        self.topo_manager.allocate_bandwidth(path, task['bw'])

        # --- 【核心修改】计算成本 ---

        # 1. 计算成本 (CPU Cost)
        # 使用动态 CPU 单价
        unit_cpu_price = self.get_dynamic_cpu_price(action_node_id)
        compute_cost = unit_cpu_price * task['cpu'] * task['duration']
        # 注意：通常云计费是 (核数 * 单价 * 时长)，这里我加上了 task['cpu'] 核数乘积，
        # 如果你的单价是"每任务"而不是"每核"，可以去掉 task['cpu']。
        # 根据表头 "每个CPU核数单位时间价格"，应该乘以核数 task['cpu']。

        # 2. 通信/带宽成本 (Bandwidth Cost)
        # 使用表中定义的带宽单价
        # 表头是 "单位时间价格"，且单位是 "带宽/Mb"
        # 假设含义是: 每 Mbps 带宽每秒的价格
        unit_bw_price = self.node_specs[action_node_id]['bw']
        comm_cost = unit_bw_price * task['bw'] * task['duration']

        total_cost = compute_cost + comm_cost

        # 奖励函数 (Cost 变小了，奖励可能会变大，需要根据实际数值范围微调 1000 这个常数)
        # 现在的 Cost 大约是:
        # CPU: 0.1 * 10核 * 20秒 = 20
        # BW:  0.3 * 20Mb * 20秒 = 120
        # 总 Cost 约 140 左右，跟之前的数值范围差不多，1000 的基准可以保留。
        reward = -total_cost + 1000

        info = {
            "status": "Success",
            "path": path,
            "cost": total_cost,
            "unit_price": unit_cpu_price  # 记录一下当前的 CPU 动态单价
        }

        return self.get_global_state(), reward, True, info

    def release_task_resources(self, task_info):
        # 保持不变
        node_id = task_info['node_id']
        cpu = task_info['cpu']
        path = task_info['path']
        bw = task_info['bw']
        self.node_resources[node_id]['used'] = max(0, self.node_resources[node_id]['used'] - cpu)
        self.topo_manager.release_bandwidth(path, bw)