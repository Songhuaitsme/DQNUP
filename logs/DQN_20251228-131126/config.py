# config.py

# ================= 拓扑与基础资源配置 =================
NODE_NUM = 12                # 节点数量 (0号为调度器，1-11为数据中心)
DEFAULT_LINK_BANDWIDTH = 500 # 默认链路总带宽 (Mbps)
DEFAULT_NODE_CPU = 80        # 默认节点总CPU资源 (单位)

# ================= 动态定价参数 =================
PRICE_ALPHA = 2.0  # 价格增长系数
PRICE_BETA = 3.0   # 指数因子 (非线性程度，反映负载对价格的敏感度)

# ================= 任务生成配置 =================
TASK_GENERATION_PROB = 0.6  # (Legacy) 单任务模式下，每个时间步生成新任务的概率

# 多任务/批处理配置
ENABLE_BATCH_TASKS = True   # 是否开启批量任务
MAX_TASKS_PER_STEP = 5      # 限制每个时间步的最大任务数 (防止显存溢出)
TASK_ARRIVAL_RATE = 2.0     # 泊松分布 Lambda 值 (平均每步到达的任务数)

# 任务属性
TASK_DURATION_MEAN = 40     # 任务平均持续时间 (时间步，指数分布)
TASK_CPU_DEMAND = (5, 20)   # 任务CPU需求范围 (min, max)
TASK_BW_DEMAND = (10, 50)   # 任务带宽需求范围 (min, max)

# ================= DQN 训练配置 =================
LEARNING_RATE = 0.001
GAMMA = 0.95               # 折扣因子
EPSILON_START = 1.0        # 初始探索率
EPSILON_MIN = 0.01         # 最小探索率
EPSILON_DECAY = 0.9999     # 探索率衰减系数

BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
MAX_STEPS = 20000          # 总训练步数

# ================= 状态与奖励配置 =================
# 状态维度: 节点位置(One-hot) + 节点负载 + 任务CPU特征 + 任务带宽特征
INPUT_DIM = NODE_NUM * 2 + 2

# 奖励缩放因子: 将较大的 Cost 数值映射到神经网络较易处理的范围
REWARD_SCALE = 1.0 / 100.0