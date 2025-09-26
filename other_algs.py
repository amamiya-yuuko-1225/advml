# class GaussianThompsonSampling(BanditAlgorithmBase):
#     def __init__(self, T, K):
#         """
#         n_arms: 臂数
#         mu0, tau0: 高斯先验 N(mu0, tau0^2)
#         sigma: 奖励噪声的标准差（假设已知）
#         """
#         self.n_arms = K
#         self.mu0 = 0.35
#         self.tau0 = 0.01
#         self.sigma2 = 0.1

#         # 每个 arm 的观测数据统计
#         self.counts = np.zeros(K)
#         self.sum_rewards = np.zeros(K)

#     def select_action(self):
#         """从每个 arm 的高斯后验分布采样"""
#         samples = []
#         for i in range(self.n_arms):
#             n = self.counts[i]
#             if n == 0:
#                 # 如果没观测过，用先验
#                 mu_n, tau_n2 = self.mu0, self.tau0**2
#             else:
#                 mean = self.sum_rewards[i] / n
#                 tau_n2 = 1.0 / (1.0/self.tau0**2 + n/self.sigma2)
#                 mu_n = tau_n2 * (self.mu0/self.tau0**2 + n*mean/self.sigma2)
#             samples.append(np.random.normal(mu_n, np.sqrt(tau_n2)))
#         return np.argmax(samples)

#     def update(self, action, reward):
#         """更新选中 arm 的观测数据"""
#         self.counts[action] += 1
#         self.sum_rewards[action] += reward

# class KLUCB_Bernoulli:
#     def __init__(self, T, K):
#         n_arms = K
#         self.n_arms = n_arms
#         self.c = 3
#         self.counts = np.zeros(n_arms, dtype=int)
#         self.rewards = np.zeros(n_arms, dtype=float)
#         self.t = 1
    
#     def kl_divergence_bernoulli(self, p, q):
#         """KL(p||q) for Bernoulli distributions."""
#         eps = 1e-15  # 避免log(0)
#         p = min(max(p, eps), 1 - eps)
#         q = min(max(q, eps), 1 - eps)
#         return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

#     def kl_ucb_upper_bound(self, p_hat, n, t, c=3, tol=1e-6):
#         """二分法求 KL-UCB 上界 (Bernoulli case)."""
#         if n == 0:
#             return 1.0  # 强制探索
        
#         rhs = (np.log(t) + c * np.log(np.log(max(t, 2)))) / n
#         low, high = p_hat, 1.0
#         while high - low > tol:
#             mid = (low + high) / 2
#             if self.kl_divergence_bernoulli(p_hat, mid) > rhs:
#                 high = mid
#             else:
#                 low = mid
#         return low


#     def update(self, arm, reward):
#         """更新奖励记录"""
#         self.counts[arm] += 1
#         self.rewards[arm] += reward
#         self.t += 1

#     def select_action(self):
#         """在第 t 轮选择 action"""
#         ucbs = []
#         for i in range(self.n_arms):
#             if self.counts[i] == 0:
#                 ucbs.append(1.0)  # 强制探索
#             else:
#                 p_hat = self.rewards[i] / self.counts[i]
#                 ucb = self.kl_ucb_upper_bound(p_hat, self.counts[i], self.t, c=self.c)
#                 ucbs.append(ucb)
#         return int(np.argmax(ucbs))

# class SuccessiveElimination(BanditAlgorithmBase):
#     def __init__(self, T, K):
#         self.n_arms = K
#         self.delta = 0.05
#         self.counts = np.zeros(K)      # 拉每个臂的次数
#         self.values = np.zeros(K)      # 每个臂的平均回报
#         self.active_arms = set(range(K))
#         self.total_pulls = 0

#     # -------------------------------
#     # Selection: 选择下一个要拉的臂
#     # -------------------------------
#     def select_action(self):
#         if len(self.active_arms) == 0:
#             raise Exception("No active arms left to select.")
#         # 随机选择一个还在候选集合中的臂
#         return np.random.choice(list(self.active_arms))

#     # -------------------------------
#     # Update: 更新平均回报并淘汰劣臂
#     # -------------------------------
#     def update(self, action, reward):
#         self.total_pulls += 1
#         self.counts[action] += 1
#         # 更新平均回报
#         n = self.counts[action]
#         self.values[action] += (reward - self.values[action]) / n

#         # 计算置信区间
#         for arm in list(self.active_arms):
#             n_arm = max(1, self.counts[arm])
#             ci = np.sqrt((1/(2*n_arm)) * np.log(4*self.n_arms*self.total_pulls**2/self.delta))
#             # 淘汰臂：如果上界小于其他臂下界
#             for other_arm in self.active_arms:
#                 if arm != other_arm:
#                     ci_other = np.sqrt((1/(2*max(1,self.counts[other_arm]))) *
#                                        np.log(4*self.n_arms*self.total_pulls**2/self.delta))
#                     if self.values[arm] + ci < self.values[other_arm] - ci_other:
#                         self.active_arms.discard(arm)
#                         break



# class ThompsonSamplingLongTail(BanditAlgorithmBase):
#     def __init__(self, T, K):
#         self.n_arms = K
#         # Gamma(alpha, beta) 先验参数（lambda 的分布）
#         self.alpha = np.ones(K)   # 初始 alpha=1
#         self.beta = np.ones(K)    # 初始 beta=1
#         self.counts = np.zeros(K)
#         self.rewards = np.zeros(K)

#     def select_action(self):
#         """
#         从 Gamma(alpha, beta) 中采样 lambda，再转成均值 1/lambda 作为效用
#         """
#         samples = np.random.gamma(self.alpha, 1.0/self.beta)
#         means = 1.0 / samples  # Exp(λ) 的均值是 1/λ
#         return np.argmax(means)

#     def update(self, action, reward):
#         """
#         reward ~ Exp(lambda) ，对应似然是：lambda * exp(-lambda * x)
#         Gamma(alpha, beta) 是其共轭先验
#         更新公式：
#             alpha' = alpha + 1
#             beta' = beta + reward
#         """
#         self.alpha[action] += 1
#         self.beta[action] += reward
#         self.counts[action] += 1
#         self.rewards[action] += reward
# class FGTS(BanditAlgorithmBase): # Feel free to update the name of the policy to better describe your solution.
#     def __init__(self, T, K):
#         """
#         Constructor of the bandit algorithm

#         Parameters
#         ----------
#         T : int
#             Horizon
#         K : int
#             Number of actions
#         """
        
#         # FILL IN CODE HERE
#         self.n_arms = K
#         self.alpha = np.ones(K) * 1
#         self.beta = np.ones(K) * 1
#         self.optimism = 0.01  # 给未选中的 arm 一点“好心情”
    
#     def select_action(self):
#         """
#         Select an action which will be performed in the environment in the 
#         current time step

#         Returns
#         -------
#         An action index (integer) in [0, K-1]
#         """
#         samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
#         return np.argmax(samples)
        
#     def update(self, action, reward):
#         """
#         Update the bandit algorithm with the reward received from the 
#         environment for the action performed in the current time step

#         Parameters
#         ----------
#         action : int
#             An action index (integer) in [0, K-1]
#         reward : int
#             Reward (integer) in {0, 1} (Bernoulli rewards)

#         """
        
#         # FILL IN CODE HERE
#         # 1) 更新选中臂：标准 TS
#         self.alpha[action] += reward
#         self.beta[action] += 1 - reward

#         # 2) 对未选中的 arms 给一点乐观更新
#         for arm in range(self.n_arms):
#             if arm != action:
#                 self.alpha[arm] += self.optimism * reward
#                 self.beta[arm] += self.optimism * (1 - reward)

# class RandUCB(BanditAlgorithmBase):
#     def __init__(self, T, K):
#         """
#         n_arms: 臂数
#         c: UCB 系数
#         noise_std: 随机扰动的标准差
#         """
#         self.n_arms = K
#         self.counts = np.zeros(K)
#         self.sums = np.zeros(K)
#         self.total_count = 0
#         self.c = 2
#         self.noise_std = 0.01

#     def select_action(self):
#         """选择动作 (RandUCB)"""
#         self.total_count += 1
#         t = self.total_count

#         # 先保证每个臂至少尝试一次
#         for i in range(self.n_arms):
#             if self.counts[i] == 0:
#                 return i

#         ucbs = np.zeros(self.n_arms)
#         for i in range(self.n_arms):
#             mean = self.sums[i] / self.counts[i]
#             bonus = self.c * np.sqrt(np.log(t) / self.counts[i])
#             noise = np.random.normal(0, self.noise_std)
#             ucbs[i] = mean + bonus + noise
#         return np.argmax(ucbs)

#     def update(self, action, reward):
#         """更新统计量"""
#         self.counts[action] += 1
#         self.sums[action] += reward


