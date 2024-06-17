#from sklearn.cluster import FuzzyCMeans
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score


# 这个函数的意义是从给定的数据路径 datapath 中加载数据，并进行以下处理：
#
# 使用 pandas 库的 read_csv 函数读取 CSV 格式的数据文件。
# 使用 sample 函数对数据进行随机打乱，以确保数据的随机性。
# 将特征部分和标签部分分别提取出来，存储在 dataX 和 labels 变量中。
# 将标签类别从字符串形式转换为数字形式，即将字符串标签 "Iris-setosa"、"Iris-versicolor" 和 "Iris-virginica" 分别映射为数字 0、1、2。
# 最终，函数返回处理后的特征数据 dataX 和对应的标签 labels。
def loadData(datapath):
    data = pd.read_csv(datapath, sep=',', header=0)
    data = data.sample(frac=1.0)   # 打乱数据顺序
    dataX = data.iloc[:, 1:-1].values # 特征
    labels = data.iloc[:, -1].values # 标签

    # 将标签类别用 0, 1, 2表示
    labels[np.where(labels == "setosa")] = 0
    labels[np.where(labels == "versicolor")] = 1
    labels[np.where(labels == "virginica")] = 2

    return dataX, labels


def loadFeatureDataAndLabels(feature_excel_path, label_excel_path):
    # 读取特征数据Excel文件
    df_features = pd.read_excel(feature_excel_path, header=None)  # 使用header=None避免将第一行数据作为列名

    # 提取设计模式名称
    design_patterns = df_features.iloc[0, 3:].values  # 假设设计模式的名称在第一行，从第四列开始

    # 提取Weight向量，假设它在第三列
    weights = df_features.iloc[1:51, 2].values  # 第三列，从第二行到第51行

    # 初始化一个空的特征数组
    features = np.zeros((50, len(design_patterns)))

    # 遍历每个设计模式的索引和名称
    for i in range(len(design_patterns)):
        # 对于每个设计模式，提取对应的特征向量（从第二行到第51行，对应列）
        pattern_features = df_features.iloc[1:51, i + 3].astype(float).values  # 调整列索引以匹配实际位置

        # 计算加权特征值
        weighted_features = pattern_features * weights

        # 将加权特征值存入特征数组
        features[:, i] = weighted_features
    # # 提取特征数据
    # features = df_features.iloc[1:51, 3:].values

    features = np.array(features,dtype=float)
    # 读取标签Excel文件
    df_labels = pd.read_excel(label_excel_path)

    # 假设标签在名为"label"的列下，且每行对应一个设计模式的标签
    labels = df_labels["label"].values
    # # 将标签类别用 0, 1, 2表示
    # labels[np.where(labels == "label_1")] = 0
    # labels[np.where(labels == "label_2")] = 1
    # labels[np.where(labels == "label_3")] = 2
    # labels[np.where(labels == "label_4")] = 3
    # labels[np.where(labels == "label_5")] = 4
    # 转换标签为数值类型
    # 创建一个映射字典，将标签字符串映射为数值
    label_mapping = {
        'label_1': 0,
        'label_2': 1,
        'label_3': 2,
        'label_4': 3,
        'label_5': 4,
    }

    # 将字符串标签转换为对应的数值标签
    labels = df_labels["label"].map(label_mapping).values

    return design_patterns, features, labels


# 这个函数的作用是初始化模糊C-均值算法中的隶属度矩阵 U。函数接受两个参数：
#
# samples：样本数量，即数据集中的样本数量。
# classes：聚类数量，即希望将数据集分成的聚类个数。
# 函数首先生成一个随机矩阵 U，其大小为 samples x classes，其中每个元素是随机生成的在 [0, 1] 范围内的值。然后，对 U 中的每一行进行归一化处理，使得每行的和为 1，即每个样本对每个聚类的隶属度之和为 1。
#
# 最终，函数返回初始化后的隶属度矩阵 U。
def initialize_U(samples, classes):
    U = np.random.rand(samples, classes)  # 先生成随机矩阵
    sumU = np.sum(U, axis=1, keepdims=True)  # 保持维度，确保sumU是列向量
    # sumU = 1 / np.sum(U, axis=1)   # 求每行的和
    # U = np.multiply(U.T, sumU)   # 使隶属度矩阵每一行和为1
    U = U / sumU  # 归一化，使隶属度矩阵每一行和为1

    return U  #这个时候U为3*150的数组

# 计算样本和簇中心的距离，这里使用欧氏距离
def distance(X, centroid):
    return np.sqrt(np.sum((X-centroid)**2, axis=1))


def computeU(X, centroids, m=2):
    sampleNumber = X.shape[0]  # 样本数
    classes = len(centroids)
    U = np.zeros((sampleNumber, classes))
    # 更新隶属度矩阵
    for i in range(classes):
        for k in range(classes):
            U[:, i] += (distance(X, centroids[i]) / distance(X, centroids[k])) ** (2 / (m - 1))
    U = 1 / U

    return U

def ajustCentroid(centroids, U, labels):
    newCentroids = [[], [], [], [], []] # 修改类别的时候需要在重新调整
    curr = np.argmax(U, axis=1)  # 当前中心顺序得到的标签
    for i in range(len(centroids)):
        index = np.where(curr == i)   # 建立中心和类别的映射
        trueLabel = list(labels[index])  # 获取labels[index]出现次数最多的元素，就是真实类别
        trueLabel = max(set(trueLabel), key=trueLabel.count)
        newCentroids[trueLabel] = centroids[i]
    return newCentroids


# def ajustCentroid(centroids, U, labels):
#     newCentroids = np.copy(centroids)  # 创建聚类中心的副本
#     curr = np.argmax(U, axis=0)  # 当前中心顺序得到的标签
#
#     for i in range(len(centroids)):
#         index = np.where(curr == i)[0]  # 建立中心和类别的映射
#
#         # 确保有属于当前中心的点
#         if len(index) > 0:
#             trueLabel = labels[index]  # 获取属于当前中心的所有标签
#
#             # 如果标签非空，则计算出现次数最多的元素
#             if trueLabel.size > 0:
#                 # 获取出现次数最多的元素
#                 most_common_label = max(set(trueLabel), key=list(trueLabel).count)
#                 newCentroids[most_common_label] = centroids[i]
#             else:
#                 # 处理没有最常见标签的情况
#                 print(f"No most common label for centroid {i}. Assigning default or skipping.")
#                 # 在这里可以选择分配一个默认中心或跳过此中心
#         else:
#             # 处理无点被分配到此中心的情况
#             print(f"No points assigned to centroid {i}. Assigning default or skipping.")
#             # 在这里可以选择分配一个默认中心或跳过此中心
#
#     return newCentroids


def cluster(data, labels, m, classes, EPS):
    """
    :param data: 数据集
    :param m: 模糊系数(fuzziness coefficient)
    :param classes: 类别数
    :return: 聚类中心
    """
    sampleNumber = data.shape[0]  # 样本数 sampleNumber= 90
    cNumber = data.shape[1]       # 特征数 cNumber = 4
    U = initialize_U(sampleNumber, classes)   # 初始化隶属度矩阵 U = 3*90
    U_old = np.zeros((sampleNumber, classes)) # U_old = 90*3
    data = np.array(data)
    U = np.array(U)
    epsilon = 1e-6

    while True:
        # centroids = np.array([])
        centroids = []
        # 更新簇中心
        for i in range(classes):
            centroid = np.dot(U[:, i]**m, data) / (np.sum(U[:, i]**m) + epsilon)
            #centroid = np.dot(U[:, i]**m, data) / (np.sum(U[:, i]**m))
            centroid = np.array(centroid, dtype=np.float64)
            centroids.append(centroid)
        # print(centroids) # 测试
        centroids = np.stack(centroids)  # 将列表转换为二维numpy数组
        U_old = U.copy()
        U = computeU(data, centroids, m)  # 计算新的隶属度矩阵

        if np.max(np.abs(U - U_old)) < EPS:
            # 这里的类别和数据标签并不是一一对应的, 调整使得第i个中心表示第i类
            centroids = ajustCentroid(centroids, U, labels)
            return centroids, U


# 预测所属的类别
def predict(X, centroids):
    labels = np.zeros(X.shape[0])
    U = computeU(X, centroids)  # 计算隶属度矩阵
    labels = np.argmax(U, axis=1)  # 找到隶属度矩阵中每行的最大值，即该样本最大可能所属类别

    return labels

# 计算内聚度
def calculate_cohesion(U, trainX, centroids):
    # 这里的计算取决于您如何定义内聚度
    # 一个简单的版本是计算每个样本到其最近聚类中心的距离的平均值
    cohesion = 0
    for i, centroid in enumerate(centroids):
        cluster_data = trainX[np.argmax(U, axis=1) == i]
        cohesion += np.sum(distance(cluster_data, centroid))
    return cohesion / len(trainX)

def calculate_separation(centroids):
    # 计算聚类中心间的平均距离
    separation = 0
    for i, centroid_i in enumerate(centroids):
        for j, centroid_j in enumerate(centroids):
            if i != j:
                separation += np.linalg.norm(centroid_i - centroid_j)
    return separation / (len(centroids) * (len(centroids) - 1) / 2)



if __name__ == '__main__':
    # 使用函数
    feature_excel_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\feature_weight_TF.xlsx" # 修改为您的特征数据Excel文件路径
    label_excel_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\data_BADP_76.xlsx"   # 修改为您的标签Excel文件路径
    design_patterns, dataX_transposed, labels = loadFeatureDataAndLabels(feature_excel_path, label_excel_path)

    dataX = np.array(dataX_transposed).T
    # print("特征数据:", dataX)
    # datapath = r"D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\Iris数据集\\iris.csv"
    # dataX, labels = loadData(datapath)  # 读取数据

    # 划分训练集和测试集
    ratio = 0.6  # 训练集的比例
    trainLength = int(dataX.shape[0] * ratio)  # 训练集长度
    trainX = dataX[:trainLength, :]
    trainLabels = labels[:trainLength]
    testX = dataX[trainLength:, :]
    testLabels = labels[trainLength:]


    EPS = 1e-6  # 停止误差条件-
    m = 2  # 模糊因子
    classes = 5  # 类别数
    # 得到各类别的中心
    centroids, U = cluster(trainX, trainLabels, m, classes, EPS)

    trainLabels_prediction = predict(trainX, centroids)
    testLabels_prediction = predict(testX, centroids)

    train_error = len(np.where((trainLabels_prediction - trainLabels) != 0)[0]) / trainLength
    test_error = len(np.where((testLabels_prediction - testLabels) != 0)[0]) / (len(dataX) - trainLength)
    print("Clustering on traintset is %.2f%%" % (train_error * 100))
    print("Clustering on testset is %.2f%%" % (test_error * 100))
    # 对于训练集
    train_misclassified = np.where(trainLabels_prediction != trainLabels)[0]
    print("训练集中被错误聚类的数据点索引:", train_misclassified)

    # 对于测试集
    test_misclassified = np.where(testLabels_prediction != testLabels)[0]
    print("测试集中被错误聚类的数据点索引:", test_misclassified)

    # 输出训练集中错误聚类的数据点的详细信息
    print("训练集中被错误聚类的数据点详细信息:")
    for idx in train_misclassified:
        print(
            f"索引: {idx}, 特征: {trainX[idx]}, 真实标签: {trainLabels[idx]}, 预测标签: {trainLabels_prediction[idx]}")

    # 输出测试集中错误聚类的数据点的详细信息
    print("\n测试集中被错误聚类的数据点详细信息:")
    for idx in test_misclassified:
        print(f"索引: {idx}, 特征: {testX[idx]}, 真实标签: {testLabels[idx]}, 预测标签: {testLabels_prediction[idx]}")

    # 计算轮廓系数
    silhouette_avg = silhouette_score(trainX, np.argmax(U, axis=1))

    # 计算调整兰德指数
    ari_score = adjusted_rand_score(trainLabels, np.argmax(U, axis=1))

    # 打印结果
    print(f"Cohesion: {calculate_cohesion(U, trainX, centroids)}")
    print(f"Separation: {calculate_separation(centroids)}")
    print(f"Silhouette Coefficient: {silhouette_avg}")
    print(f"Adjusted Rand Index: {ari_score}")



