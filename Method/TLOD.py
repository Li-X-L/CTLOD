# TLOD.py
"""
时序局部离群点检测
方案：近邻特征提取+序列邻域+多序列近邻离群因子
修改：更改距离计算方法，均值改为密度小值
"""

import numpy as np
import pandas as pd


class SLOF:
    """
    局部离群因子(LOF)算法实现
    """
    def __init__(self, k=3, n_neighbors=12, rd_method="mean", lof_method='mean'):
        """
        局部离群因子(LOF)算法实现
        参数:
            k: 序列近邻距离特征集近邻值
            n_neighbors: 用于计算局部密度的邻居数量
            rd_method: 近邻密度邻域距离计算方法，
                <mean>：均值；
                <median>：中值
            lof_method: 基于各k值结果进行lof计算方法
                <mean>：均值；
                <max>：最大值
        """
        self.n_neighbors = n_neighbors
        self.k = k
        self.rd_method = rd_method
        self.lof_method = lof_method
        self.neighbors = None
        self._distance_lower_line = None
        self.threshold = None
        self.lrd = None
        self.lof_scores = None

    def fit_predict(self, x, contamination= None, distance_lower_line = 1e-8):
        """
        拟合模型并计算每个样本的LOF分数

        :parameter
            x: 输入数据，形状为(n_samples, n_features)
            contamination: 污染率
                -float(>=1)，表示阈值
                -float（0-1），表示异常比例
            distance_lower_line: 近邻距离下限，默认设为1e-8，避免除零
        return:
            lof_scores: 每个样本的LOF分数
            outliers: 每个样本的异常标识
        """
        self._distance_lower_line = distance_lower_line
        # 转换为numpy数组
        x = np.array(x)
        n_samples = x.shape[0]

        # 确保邻居数量小于样本数量
        if self.n_neighbors >= n_samples:
            raise ValueError("n_neighbors必须小于样本数量")

        # 计算每个点的k个最近邻
        self.neighbors = self._find_neighbors(data = x, num_neighbors = self.n_neighbors)

        # 特征提取
        feature_x = self._feature_extrac(x)

        # 计算每个点的局部可达密度
        self.lrd = self._local_reachability_density(feature_x)

        # 计算每个点的LOF分数
        self.lof_scores = self._compute_lof(self.lrd)

        # 计算离群点，根据LOF分数
        lof_outliers = np.zeros_like(self.lof_scores, dtype=int)
        if contamination is not None:
            if contamination > 1:
                self.threshold = contamination
            elif 0 < contamination < 1:
                self.threshold = np.percentile(self.lof_scores, (1-contamination)*100)
            elif contamination == 0:
                self.threshold = 3
            else:
                raise ValueError("contamination设置错误！")
        else:
            self.threshold = 3
        lof_outliers[np.where(self.lof_scores > self.threshold)] = 1

        return lof_outliers, self.lof_scores

    def _feature_extrac(self, data):
        """
        提取原始单变量数据特征
        :parameter
            data: numpy数组
        return:
            feature_x: numpy，包括每一点不同间隔k，差分数据集，前后补nan，
            每列前补nan为列值，(n_samples, n+k)
        """

        feature_x = []

        for interval in range(1, self.k+1):

            interval_diff = [np.nan]*interval
            value = np.abs(data[interval:] - data[: -interval])

            # 整合
            feature_x.append(np.concatenate([interval_diff, value, interval_diff]))

        df = pd.DataFrame(feature_x)

        return df.to_numpy().T

    def _find_neighbors(self, data, num_neighbors):
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

    def _local_reachability_density(self, feature_x):
        """计算每个点的局部可达密度"""

        n_samples = self.neighbors.shape[0]
        lrd = np.zeros(shape = (n_samples, self.k), dtype=float)

        for i in range(n_samples):

            neighbors_set_i = self.neighbors[i]
            temp_mean = []
            temp_median = []
            d_k_p_l = []

            for j in range(self.k):
                # 获取i点邻居的特征（相关近邻距离）索引集
                feature_neighbors_set_i = np.union1d(neighbors_set_i, neighbors_set_i+j+1)
                for delete_index in [i, i+j+1]:
                    index = np.where(feature_neighbors_set_i == delete_index)
                    feature_neighbors_set_i = np.delete(feature_neighbors_set_i, index)

                # 获得 (k,n_neighbors)近邻特征数据集

                temp_mean.append(np.nanmean(feature_x[feature_neighbors_set_i, j]))     #  前后近邻距离均值
                temp_median.append(np.nanmedian(feature_x[feature_neighbors_set_i, j]))

                # 获得 (k,1)近邻特征数据集

                d_k_p_l.append(np.nanmin(feature_x[[i, i+j+1], j]))      #  前后近邻距离较小值
            # 将列表转为数组
            d_k_p = np.array(d_k_p_l)

            if self.rd_method == "mean":
                # 近邻k距离均值计算，（k,1）
                k_distances_t = np.array(temp_mean)
                k_distances = np.where(k_distances_t < self._distance_lower_line, self._distance_lower_line, k_distances_t)
            elif self.rd_method == "median":
                # 近邻k距离中位数计算，（k,1）
                k_distances_t = np.array(temp_median)
                k_distances = np.where(k_distances_t < self._distance_lower_line, self._distance_lower_line, k_distances_t)
            else:
              raise ValueError("rd_method参数设置有误")

            # 近邻距离密度（1,k）
            lrd[i] = d_k_p / k_distances

        return lrd

    def _compute_lof(self, lrd):
        """计算每个点的LOF分数"""

        if self.lof_method == "mean":
            lof = np.nanmean(lrd, axis=1)
        elif self.lof_method == "max":
            lof = np.nanmax(lrd, axis=1)
        else:
            raise ValueError("lof_method参数设置有误")

        return lof


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
    lof = SLOF(4, 16, rd_method="mean", lof_method="mean")
    out, lof_scores = lof.fit_predict(X, contamination=0.05)

    # 输出结果
    print("前10个正常点的LOF分数:")
    print(sorted(lof_scores)[:10])
    print("\n离群点的LOF分数:")
    print(sorted(lof_scores, reverse=True)[:10])

    outliers = np.where(out ==1)[:10]
    print("\n检测到的离群点索引:", outliers)
