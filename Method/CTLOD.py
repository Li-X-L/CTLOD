# CTLOD.py
"""
时序局部选择性组合并行离群集成
"""

import numpy as np
import pandas as pd
from Method.SLOF import SLOF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _find_neighbors(data, num_neighbors):
    """
    找到每个点的k个最近邻及其距离
    :parameter
        data: numpy，数据集
        num_neighbors: 近邻数
    :return
        windows_neighbors: 时序数据集每个点的近邻数据索引
        neighbors_set: numpy，每个点的近邻距离数据集
    """

    n_timesteps = len(data)

    # 初始化窗口结果
    windows_neighbors = np.zeros((n_timesteps, num_neighbors), dtype=int)
    indexes = np.arange(0, n_timesteps)

    # 从第k个时间点开始（索引从0开始，需至少有k个历史数据）
    for t in range(n_timesteps):
        # 窗口数据
        if t - num_neighbors//2 < 0:
            windows_neighbors[t] = np.concatenate([indexes[: t], indexes[t+1 : num_neighbors+1]])
        elif t + num_neighbors//2 >= n_timesteps:
            windows_neighbors[t] = np.concatenate([indexes[n_timesteps-num_neighbors-1 : t], indexes[t+1 : n_timesteps]])
        else:
            windows_neighbors[t] = np.concatenate([indexes[t - num_neighbors // 2: t], indexes[t + 1: t + num_neighbors // 2 + 1]])

    return windows_neighbors


class CSLOF:
    """
    局部离群因子集成算法实现
    """
    def __init__(self, k_set, n_neighbors_set,
                 rd_method, lof_method, local_region_size=20,
                 n_bins=5, std_method = 0, com_method = "aom"):
        """
        序列LSCP算法实现
        参数:
            k_set: 序列近邻距离特征集近邻值，SLOF
            n_neighbors_set: 用于计算局部密度的邻居数量，SLOF
            rd_method: 近邻密度邻域距离计算方法，
                <mean>：均值；
                <median>：中值
            lof_method: 基于各k值结果进行lof计算方法
                <mean>：均值；
                <max>：最大值
            n_neighbors: SLSCP邻居参数
            std_method: 归一化方法
            com_method: 组合方法
        """
        self.n_neighbors_set = n_neighbors_set
        self.k_set = k_set
        self.rd_method = rd_method
        self.lof_method = lof_method
        self.local_region_size = local_region_size
        self.n_bins = n_bins
        self.std_method = std_method
        self.com_method = com_method
        self.neighbors = None
        self.contamination = None
        self.threshold = None
        self.threshold_set = None
        self.scores_set = None
        self.scores = None
        self.outliers = None

    def fit_predict(self, x, contamination= None, distance_lower_line = 1e-8):
        """
        计算每组参数下每个样本的SLOF分数，形成基检测器集

        :parameter
            x: 输入数据，形状为(n_samples, n_features)
            contamination: 污染率
                -float(>=1)，表示阈值
                -float（0-1），表示异常比例
            distance_lower_line: 近邻距离下限，默认设为1e-8，避免除零
        return:
            self.scores：集成分数
        """
        self.contamination = contamination

        # 基检测器集结果
        self.scores_set = self._get_scores(x, distance_lower_line)

        # 近邻集获取
        self.neighbors = _find_neighbors(data = x, num_neighbors = self.local_region_size)

        # 动态选择组合并行集成结果
        self.scores = self._lscp()
        self.outliers = np.zeros_like(self.scores, dtype=int)
        if contamination is not None:
            if 0 < contamination < 1:
                self.threshold = np.percentile(self.scores, (1-contamination)*100)
            elif contamination == 0:
                self.threshold = 3
            elif contamination > 1:
                self.threshold = contamination
            else:
                raise ValueError("contamination设置错误！")
        else:
            self.threshold = 3
        self.outliers[np.where(self.scores >= self.threshold)] = 1

        return self.outliers, self.scores

    def _get_scores(self, data, distance_lower_line):
        """
        计算每组参数下每个样本的SLOF分数，形成基检测器集
        标准化、归一化、阈值归一化
        :parameter
            data: numpy数组
            contamination: 污染率
                -float(>=1)，表示阈值
                -float（0-1），表示异常比例
            distance_lower_line: 近邻距离下限，默认设为1e-8，避免除零
        return:
            scores: numpy，处理后数据
        """

        # 输出列表初始化
        raw_scores = []
        threshold_set = []
        for i_k in self.k_set:
            for i_n in self.n_neighbors_set:
                for i_rd in self.rd_method:
                    for i_lof in self.lof_method:
                        lof = SLOF(k = i_k, n_neighbors = i_n, rd_method= i_rd, lof_method=i_lof)
                        _, lof_scores = lof.fit_predict(data, self.contamination, distance_lower_line)
                        raw_scores.append(lof_scores)
                        threshold_set.append(lof.threshold)

        scores_set= np.array(raw_scores).T
        self.threshold_set = np.array(threshold_set)

        return scores_set

    def _lscp(self):
        """
        计算并行离群检测的局部选择集成
        """

        n_samples = self.neighbors.shape[0]
        pred_scores_ens = np.zeros([n_samples, ])
        # 归一化（全局）
        scores_set_scaler = self._scores_scaler(self.scores_set)

        for i in range(n_samples):
            # 邻居数据集
            neighbors_set_i = self.neighbors[i]
            scores_set_i = scores_set_scaler[neighbors_set_i, :]

            if self.com_method == "aom":
                pseudo_target = np.max(scores_set_i, axis=1)
                # 计算 Pearson 相关系数
                df = pd.concat([pd.DataFrame(pseudo_target), pd.DataFrame(scores_set_i)], axis=1)
                corr_matrix = df.corr().values
                pearson_corr = corr_matrix[0, 1:]
                # 动态选择
                pred_scores_ens[i, ] = np.mean(scores_set_scaler[i, self._get_competent_detectors(pearson_corr)])
            elif self.com_method == "moa":
                pseudo_target = np.mean(scores_set_i, axis=1)
                # 计算 Pearson 相关系数
                df = pd.concat([pd.DataFrame(pseudo_target), pd.DataFrame(scores_set_i)], axis=1)
                corr_matrix = df.corr().values
                pearson_corr = corr_matrix[0, 1:]
                # 动态选择
                pred_scores_ens[i, ] = np.max(scores_set_scaler[i, self._get_competent_detectors(pearson_corr)])
            elif self.com_method == "a":
                pred_scores_ens[i,] = np.mean(scores_set_scaler[i, :])
            elif self.com_method == "m":
                pred_scores_ens[i,] = np.max(scores_set_scaler[i, :])
            else:
                raise ValueError("com_method 输入错误！")

        return pred_scores_ens

    def _scores_scaler(self, data):
        """
        标准化、归一化、阈值归一化
        :parameter
            data: numpy数组
        return:
            scores: numpy，处理后数据
        """

        if self.std_method == 0:
            scores_scaler = StandardScaler().fit_transform(data)
        elif self.std_method == 1:
            scores_scaler = MinMaxScaler().fit_transform(data)
        elif self.std_method == 2:
            temp = data.max(axis=0) - data.min(axis=0)
            temp[temp == 0] = 1e-3
            scores_scaler = (data - self.threshold_set) / temp
        else:
            scores_scaler= data

        return scores_scaler


    def _get_competent_detectors(self, scores):
        """ Identifies competent base detectors based on correlation scores

        Parameters
        ----------
        scores : numpy array,
            Correlation scores for each classifier (for a specific
            test instance)

        Returns
        -------
        candidates : List
            Indices for competent detectors (for given test instance)
        """

        # create histogram of correlation scores
        scores = scores.reshape(-1, 1)

        # if scores contain nan, change it to 0
        if np.isnan(scores).any():
            scores = np.nan_to_num(scores)

        hist, bin_edges = np.histogram(scores, bins=self.n_bins)

        max_value = np.max(hist)
        max_bins = np.where(hist == max_value)[0]

        candidates = []

        # iterate through bins
        for max_bin in max_bins:
            # determine which detectors are inside this bin
            selected = np.where((scores >= bin_edges[max_bin])
                                & (scores <= bin_edges[max_bin + 1]))

            # add to list of candidates
            candidates = candidates + selected[0].tolist()

        return candidates



# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    # np.random.seed(42)
    # # 正常数据点（二维高斯分布）
    # X_normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    # # 离群点
    # X_outliers = np.random.uniform(low=-6, high=6, size=(5, 2))
    # # 合并数据
    # X = np.vstack([X_normal, X_outliers])
    # X = np.array(np.random.randint(0, 1000, 100)).T
    X1 = np.array(range(50))
    X2 = np.ones(100)*55
    X3 = np.array(range(50, 0, -1))
    X = np.concatenate((X1, X2, X3))
    indices = [10, 20, 21, 40, 70, 71, 72, 90, 120, 140]
    X[indices] = [15, 30, 30, 45, 45, 47, 45, 60, 45, 15]

    # 应用LOF算法
    # lof = SLOF(4, 16, rd_method="mean", lof_method="mean")
    # lof_scores, out = lof.fit_predict(X, contamination=0.05, distance_lower_line=None)
    contamination = 0.05
    slscp = CSLOF(k_set = [3,4,5], n_neighbors_set = [10,20,30], rd_method = ["mean","median"], lof_method = ["mean","max"]
                  , local_region_size = 20, n_bins = 5, std_method =0, com_method = "aom")
    outliers, scores = slscp.fit_predict(X, contamination, distance_lower_line=1)

    # 输出结果
    print("前10个正常点的LOF分数:")
    print(sorted(scores)[:10])
    print("\n离群点的LOF分数:")
    print(sorted(scores, reverse=True)[:10])

    outliers = np.where(outliers == 1)
    # outliers = np.where(scores > 0)
    print("\n检测到的离群点索引:", outliers)
