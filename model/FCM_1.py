import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from weight_assignment import *
from numpy import dot
from numpy.linalg import norm
from preprocess import *

def loadFeatureDataAndLabels(feature_excel_path, label_excel_path):
    # 读取特征数据Excel文件
    df_features = pd.read_excel(feature_excel_path, header=None)  # 使用header=None避免将第一行数据作为列名

    # 提取设计模式名称
    design_patterns = df_features.iloc[0, 3:].values  # 假设设计模式的名称在第一行，从第四列开始

    # 提取Weight向量，假设它在第三列
    weights = df_features.iloc[1:301, 2].values  # 第三列，从第二行到第51行 修改特征值

    # 初始化一个空的特征数组
    features = np.zeros((300, len(design_patterns))) # 修改特征值

    # 遍历每个设计模式的索引和名称
    for i in range(len(design_patterns)):
        # 对于每个设计模式，提取对应的特征向量（从第二行到第51行，对应列）
        pattern_features = df_features.iloc[1:301, i + 3].astype(float).values  # 调整列索引以匹配实际位置  修改特征值

        # 计算加权特征值
        weighted_features = pattern_features * weights

        # 将加权特征值存入特征数组
        features[:, i] = weighted_features
    # # 提取特征数据
    # features = df_features.iloc[1:51, 3:].values

    features = np.array(features, dtype=float)
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


from scipy.spatial.distance import cdist


from scipy.spatial.distance import cdist

def initialize_membership_matrix(n_samples, n_clusters):
    """ Randomly initialize the membership matrix for FCM. """
    membership_mat = np.random.random((n_samples, n_clusters))
    membership_mat = membership_mat / np.sum(membership_mat, axis=1)[:, np.newaxis]
    return membership_mat

def calculate_cluster_centers(X, membership_mat, m):
    """ Calculate cluster centers from the membership matrix and data points. """
    cluster_mem_val = np.power(membership_mat, m)
    cluster_centers = (X.T @ cluster_mem_val / np.sum(cluster_mem_val, axis=0)).T
    return cluster_centers

# 余弦相似度距离
def update_membership_matrix(X, cluster_centers, m):
    """Update the membership matrix based on the data points and current cluster centers using cosine distance."""
    n_samples = X.shape[0]
    n_clusters = cluster_centers.shape[0]
    new_membership_mat = np.zeros((n_samples, n_clusters))

    # 使用余弦距离进行计算
    power = 2 / (m - 1)
    denominator = cdist(X, cluster_centers, 'cosine')  # 使用'cosine'而不是默认的'euclidean'
    denominator = np.power(denominator, power)
    denominator[denominator == 0] = np.finfo(X.dtype).eps  # Avoid division by zero

    for i in range(n_clusters):
        denominator[:, i] = np.sum(1.0 / denominator, axis=1)

    new_membership_mat = 1 / denominator
    return new_membership_mat
# 欧式距离
# def update_membership_matrix(X, cluster_centers, m):
#     """ Update the membership matrix based on the data points and current cluster centers. """
#     n_samples = X.shape[0]
#     n_clusters = cluster_centers.shape[0]
#     new_membership_mat = np.zeros((n_samples, n_clusters))
#
#     power = 2 / (m - 1)
#     denominator = cdist(X, cluster_centers, 'euclidean')
#     denominator = np.power(denominator, power)
#     denominator[denominator == 0] = np.finfo(X.dtype).eps  # Avoid division by zero
#
#     for i in range(n_clusters):
#         denominator[:, i] = np.sum(1.0 / denominator, axis=1)
#
#     new_membership_mat = 1 / denominator
#     return new_membership_mat

def fuzzy_c_means_clustering(X, n_clusters, m, error, maxiter):
    """ Perform Fuzzy C-Means clustering. """
    n_samples = X.shape[0]
    membership_mat = initialize_membership_matrix(n_samples, n_clusters)
    curr = 0
    while curr <= maxiter:
        cluster_centers = calculate_cluster_centers(X, membership_mat, m)
        new_membership_mat = update_membership_matrix(X, cluster_centers, m)
        if np.linalg.norm(new_membership_mat - membership_mat) < error:
            break
        membership_mat = new_membership_mat
        curr += 1

    labels = np.argmax(membership_mat, axis=1)
    return labels, cluster_centers, membership_mat
def compute_distance_to_centroids(X, centroids):
    """
    Compute the Euclidean distance from each sample to each cluster center.

    Parameters:
    X - array-like of shape (n_samples, n_features)
    centroids - array-like of shape (n_clusters, n_features)

    Returns:
    distances - array-like of shape (n_samples, n_clusters)
    """
    # Use cdist function from scipy to compute distances
    distances = cdist(X, centroids, metric='cosine')  # euclidean 欧式距离
    return distances


def assign_clusters(distances):
    """
    Assign samples to the nearest cluster based on the distance matrix.

    Parameters:
    distances - array-like of shape (n_samples, n_clusters)

    Returns:
    labels - array-like of shape (n_samples,)
    """
    labels = np.argmin(distances, axis=1)
    return labels


def calculate_feature_vector(excel_path, word_list, num_features):
    # 读取Excel文件，跳过包含非数值数据的头部行（如果有的话）
    df_features = pd.read_excel(excel_path, header=None)

    # 假设第一行是列名（header），我们跳过这一行，并从第二行开始读取数据
    df_features = df_features[1:]

    # 创建一个零初始化的numpy数组
    feature_vector = np.zeros((len(df_features),), dtype=int)

    # 迭代单词列表，如果单词在Excel中，将相应位置设为1
    for word in word_list:
        # 注意这里我们假设第一列（index 0）是包含单词的列
        if word in df_features.iloc[:, 1].values:  # 现在第一行是数据
            word_indices = df_features.index[df_features.iloc[:, 1] == word].tolist()
            # 设置对应单词位置的特征为1
            feature_vector[np.array(word_indices) - df_features.index.start] = 1

    # 获取权重列，并转换为浮点数
    weights = df_features.iloc[:num_features, 2].astype(float).values

    # 计算加权特征向量
    weighted_feature_vector = feature_vector[:num_features] * weights

    # 将结果重塑为1行的数组形式，以便后续处理
    weighted_feature_vector = weighted_feature_vector.reshape(1, -1)

    return weighted_feature_vector




def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    :param vector_a: A numpy array or a list representing the first vector.
    :param vector_b: A numpy array or a list representing the second vector.
    :return: Cosine similarity between the two vectors.
    """
    # Convert lists to numpy arrays if necessary
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    # Compute the cosine similarity
    cos_sim = dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))
    return cos_sim


def normalize_vector(vector):
    """Normalize a vector to a unit vector."""
    return vector / norm(vector)


def cosine_distance_to_all_centers(feature_vector, cluster_centers):
    """
    Calculate the cosine distance of a feature vector to all cluster centers
    and convert it to similarity measure.

    :param feature_vector: A numpy array representing the feature vector.
    :param cluster_centers: A numpy array representing the cluster centers.
    :return: Normalized similarity measures to all cluster centers.
    """
    # Normalize the feature vector and cluster centers to unit vectors
    feature_vector_norm = normalize_vector(feature_vector)
    cluster_centers_norm = np.array([normalize_vector(center) for center in cluster_centers])

    # Calculate cosine similarity
    cosine_similarities = np.dot(cluster_centers_norm, feature_vector_norm)

    # Convert similarities to distances
    cosine_distances = 2 - cosine_similarities

    # Normalize distances to sum up to 1
    normalized_distances = cosine_distances / np.sum(cosine_distances)

    return normalized_distances


def pca_transform_and_separate(dataX, weighted_feature_vector, n_components=50):
    # 确保weighted_feature_vector是行向量
    weighted_feature_vector_2d = np.atleast_2d(weighted_feature_vector)

    # 检查weighted_feature_vector是否需要转置
    if weighted_feature_vector_2d.shape[0] != 1:
        weighted_feature_vector_2d = weighted_feature_vector_2d.T

    # 将dataX和weighted_feature_vector沿着特征轴拼接
    combined_data = np.concatenate((dataX, weighted_feature_vector_2d), axis=0)

    # 初始化PCA对象
    pca = PCA(n_components=n_components)

    # 对拼接的数据进行PCA降维
    combined_data_reduced = pca.fit_transform(combined_data)

    # 分离降维后的数据为dataX_reduced和weighted_feature_vector_reduced
    dataX_reduced = combined_data_reduced[:-1, :]  # 所有行除了最后一行
    weighted_feature_vector_reduced = combined_data_reduced[-1, :]  # 最后一行

    return dataX_reduced, weighted_feature_vector_reduced

def find_closest_labels(normalized_distances, threshold=0.01):
    # 使用numpy找到最大距离值及其索引
    max_distance = np.max(normalized_distances)
    max_index = np.argmax(normalized_distances)

    # 初始化一个列表来存储最大值和接近最大值的标签
    closest_labels = [max_index]

    # 遍历距离数组，找出与最大值相近的索引
    for index, distance in enumerate(normalized_distances):
        if index != max_index and max_distance - distance <= threshold:
            closest_labels.append(index)

    return closest_labels


def find_design_patterns(dataX_reduced, closest_labels, assigned_clusters, weighted_feature_vector_reduced):
    """
    Find design patterns belonging to the closest labels and calculate the cosine similarity with the weighted feature vector.

    :param dataX_reduced: The reduced feature matrix of design patterns.
    :param closest_labels: The closest labels determined by the threshold.
    :param assigned_clusters: The assigned clusters for each design pattern.
    :param weighted_feature_vector_reduced: The reduced weighted feature vector for the design problem.
    :return: Indices of design patterns belonging to the closest labels and their cosine similarity values.
    """
    # Find indices of design patterns that belong to the closest labels
    pattern_indices = np.where(np.isin(assigned_clusters, closest_labels))[0]

    # Calculate cosine similarity for these design patterns
    similarity_values = []
    for index in pattern_indices:
        similarity = cosine_similarity(weighted_feature_vector_reduced, dataX_reduced[index])
        similarity_values.append(similarity)

    return pattern_indices, similarity_values


def filter_by_threshold(similarity_values, threshold):
    # 将列表转换为 numpy 数组
    similarity_values_array = np.array(similarity_values)
    # 使用 numpy 数组进行比较
    filtered_indices = np.where(similarity_values_array > threshold)[0]
    filtered_values = similarity_values_array[filtered_indices]
    return filtered_indices, filtered_values

def compare_with_max(similarity_values, max_similarity_value, threshold):
    # 将列表转换为 numpy 数组
    similarity_values_array = np.array(similarity_values)
    # 进行计算和筛选
    diff = max_similarity_value - similarity_values_array
    valid_indices = np.where(diff < threshold)[0]
    valid_values = similarity_values_array[valid_indices]
    return valid_indices, valid_values


if __name__ == '__main__':
    np.random.seed(6)  # 设置随机数种子
    # 使用函数
    feature_excel_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\feature_weights_LB.xlsx" # 修改为您的特征数据Excel文件路径
    label_excel_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\data_BADP_76.xlsx"   # 修改为您的标签Excel文件路径
    design_patterns, dataX_transposed, labels = loadFeatureDataAndLabels(feature_excel_path, label_excel_path)
    dataX = np.array(dataX_transposed).T
    # text = ""
    # text = "A set of low-value payments (aka., micropayments) are to be made frequently, e.g., a small payment paid every time a WiFi service is used.If a public blockchain is used, micro-payments are too expensive to make as the transaction fee might be higher than the monetary value within the transaction. Moreover, it takes time to achieve finality on a blockchain. Furthermore, as public blockchains have limited performance, transactions can take several minutes or even an hour to be committed on the blockchain. Therefore, how can frequent payments be processed without waiting for a long time and incurring high transaction fees?"
    # text = "In the context of blockchain-based payments – Once the sellers receive tokens from the buyers, they need to redeem the tokens for fiat currency (e.g., US Dollars). Once the tokens are redeemed in return for a fiat currency payment, they should not be further usable (i.e., prevent double-spending).In the context of data migration – The source blockchain is public and not decommissioned after the data migration. Therefore, any state and smart contracts left in the source blockchain could be misused (e.g., double spending). The list of states and smart contracts to be migrated is given in the snapshot or application-level reference to the blockchain identifier mapping database.How to ensure tokens, states, and smart contracts are unusable when they are no longer required?"
    # text = "Usually, an identification process lasts for a certain time period. After proving the identity of a party, the presented credential has accomplished its mission and should not be accessed again.Note that the narrative of this pattern is presented in the context of self-sovereign identities. An identity credential could be generalised to any digital or digitalised content stored on the blockchain.After receiving a credential, a verifier can continue to access, read, and verify certain identity data of the holder. If the credential is long-term or even permanently effective, the verifier can check the holder’s identity data even when there is no legitimate reason to do so. How to provide access to a credential only once?"
    text = "The integrity of a large datum or a large collection of data (that may not fit onto a blockchain transaction) or dynamic data needs to be preserved.The blockchain, due to its full replication across all participants of the blockchain network, has limited storage capacity. Storing large volumes of data within a transaction may be impossible due to the limited size of the transaction and blocks of the blockchain. For example, Ethereum has a block gas limit to determine the number, computational complexity, and data size of the transactions included in the block. Also, the throughput could be limited. Data cannot take advantage of the immutability or integrity guarantees without being stored on the blockchain. How to preserve the integrity of a large set of data or dynamic data?"
    # text = "Due to forking, the immutability of a blockchain using Nakamoto consensus is probabilistic. Due to the longest chain wins rule of the Nakamoto consensus, the current longest chain of blocks could be overtaken by another branch of blocks with non-zero probability. Then the transactions included in the current chain of blocks is no longer considered valid. Hence, there is always a chance that the most recent blocks are replaced by a competing chain fork. The transactions that were included in those blocks are discarded and eventually go back to the transaction pool and are considered for inclusion in a later block.When a fork occurs, it is uncertain as to which branch will be permanently kept in the blockchain and which branch(es) will be discarded. Therefore, the inclusion of a transaction in a block is insufficient to declare it as confirmed (or durable in terms of ACID properties). How can we ensure that a transaction is permanently included in a block?"

    # text = "Like any software application, a smart contract may need to be eventually upgraded to fix bugs, overcome security weaknesses, or add new functionality. In general, business logic and data changes are required at different times and frequencies."
    text_dict_1 = preprocess_sent_1(text)
    text_dict_2 = preprocess_word_1(text_dict_1)
    # print(text_dict_2)
    excel_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\feature_weights_LB.xlsx"
    weighted_feature_vector = calculate_feature_vector(excel_path, text_dict_2, 300) # 修改特征值
    # print(weighted_feature_vector)
    # 选择降维后的特征数量，例如降到100维
    dataX_reduced, weighted_feature_vector_reduced = pca_transform_and_separate(dataX, weighted_feature_vector,n_components=2)

    # 初始化 silhouette_avg 使得循环可以开
    silhouette_avg = 0

    # 循环直到 silhouette_avg * 2 ≥ 1
    while silhouette_avg * 2 < 1.33: #最高1.5
        fcm_labels, fcm_centers, fcm_membership = fuzzy_c_means_clustering(dataX_reduced, n_clusters=5, m=2.4,
                                                                           error=0.0001, maxiter=1000)
        distances = compute_distance_to_centroids(dataX_reduced, fcm_centers)
        assigned_clusters = assign_clusters(distances)
        silhouette_avg = silhouette_score(dataX_reduced, assigned_clusters)
        print("The average silhouette_score is :", silhouette_avg * 2)
    # fcm_labels, fcm_centers, fcm_membership = fuzzy_c_means_clustering(dataX_reduced, n_clusters=5, m=300, error=0.000001, maxiter=100000)
    # # fcm_labels, fcm_centers, fcm_membership = fuzzy_c_means_clustering(dataX, n_clusters=5, m=2, error=0.000001,maxiter=100000)
    # # Now, let's calculate distances and assign clusters
    # distances = compute_distance_to_centroids(dataX_reduced, fcm_centers) #dataX_reduced
    # assigned_clusters = assign_clusters(distances)
    # # 在完成Fuzzy C-Means聚类后，计算轮廓系数
    # silhouette_avg = silhouette_score(dataX_reduced, assigned_clusters) #dataX_reduced
    # print("The average silhouette_score is :", silhouette_avg*2)

    # Display the results
    print("FCM cluster centers:")
    print(fcm_centers)

    print("\nTrue labels:")
    print(labels)
    # Output the assigned clusters and distances for inspection
    print("Assigned clusters for each sample:")
    print(assigned_clusters)

    print("\nDistance of each sample to each cluster center:")
    for i, distance_to_centers in enumerate(distances):
        print(f"Sample {i}: {distance_to_centers}")


    normalized_distances = cosine_distance_to_all_centers(weighted_feature_vector_reduced, fcm_centers) #weighted_feature_vector_reduced
    print(normalized_distances)

    closest_labels = find_closest_labels(normalized_distances)
    print("Closest labels within the threshold:", closest_labels)
    pattern_indices, similarity_values = find_design_patterns(dataX, closest_labels, assigned_clusters, weighted_feature_vector)
    print(pattern_indices)
    print(similarity_values*2)
    # 找出最相似的设计模式的索引
    max_similarity_index = np.argmax(similarity_values)

    # 找出最相似的设计模式的相似度值
    max_similarity_value = similarity_values[max_similarity_index]

    print("Most similar design pattern index:", max_similarity_index)
    print("Highest similarity value:", max_similarity_value*2)


    filtered_indices, filtered_values = filter_by_threshold(similarity_values, 0.2)
    print("Filtered by threshold:")
    print("Indices:", filtered_indices)
    print("Values:", filtered_values*2)

    # Compare with max similarity
    compared_indices, compared_values = compare_with_max(similarity_values, max_similarity_value, 0.01)
    print("\nCompared with max similarity:")
    print("Indices:", compared_indices)
    print("Values:", compared_values*2)