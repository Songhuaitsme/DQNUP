# --- 拓扑配置 ---
NODE_NUM = 12  # 0号为调度器，1-11为数据中心
DEFAULT_LINK_BANDWIDTH = 500  # 默认链路总带宽 (Mbps)
DEFAULT_NODE_CPU = 80        # 默认节点总CPU资源 (单位)
# --- 动态定价参数 ---
PRICE_ALPHA = 2.0  # 价格增长系数
PRICE_BETA = 3.0   # 指数因子 (非线性程度)
# --- 任务配置 ---
TASK_GENERATION_PROB = 0.6  # 每个时间步生成新任务的概率
#新增-多任务到达逻辑===================================
ENABLE_BATCH_TASKS = True   # 开关
MAX_TASKS_PER_STEP = 5      # 每个时间步最多来多少个任务
# 或者使用泊松分布的 lambda (平均每步来多少任务)
TASK_ARRIVAL_RATE = 2.0
#==================================================
TASK_DURATION_MEAN = 40    # 任务平均持续时间 (时间步)
TASK_CPU_DEMAND = (5, 20)   # 任务CPU需求范围 (min, max)
TASK_BW_DEMAND = (10, 50)   # 任务带宽需求范围 (min, max)
# --- DQN 训练配置 ---
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
MAX_STEPS = 10000  # 总模拟步数
# =======================================================
# --- 修改点 1: 状态维度 ---
# 原来是 NODE_NUM * 2，现在加上 task_cpu 和 task_bw 两个特征
INPUT_DIM = NODE_NUM * 2 + 2

# --- 修改点 2: 奖励缩放 (新增) ---
# 用来把巨大的 Cost 缩小到神经网络喜欢的范围 (-1 到 1 左右)
REWARD_SCALE = 1.0 / 100.0