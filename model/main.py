from preprocess import *
from LDA_Gibbs import *
from reader import *
import warnings
import BERT
from autoencoder import *
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
import numpy as np
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    rare_words_file = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\blockchain_word.txt"
    excel_file = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\data_DPP-review.xlsx"  # Excel文件路径
    column_title = "raw_data"  # 指定的列标题
    text_data = read_text_column_from_excel(excel_file, column_title)
    # 使用字典来存储文本和对应的标签
    text_label_dict = []
    embedding_array = []
    successful_texts = []  # 用于存储能够被BERT处理的文本
    for text in text_data:
        # 确保输入是单个文本字符串或字符串列表
        if isinstance(text, str):
            text_dict = {"text": text, "label": None}  # 初始时，标签设为None
            text_label_dict.append(text_dict)
        elif isinstance(text, list) and all(isinstance(item, str) for item in text):
            text_dict = {"text": text, "label": None}  # 初始时，标签设为None
            text_label_dict.append(text_dict)

        else:
            # 如果不是字符串或字符串列表，则跳过这个文本
            continue
        # 对每个文本使用 BERT.text_to_bert_embedding 函数进行处理
        successful_texts.append(text)
        embedding = BERT.text_to_bert_embedding(text)
        # 将处理后的嵌入添加到 embedding_array 数组中
        embedding_array.append(embedding)


    #文本预处理
    pre_text_data = preprocess_sent(successful_texts) #文本级处理
    pre_text_word = preprocess_word(pre_text_data) #词级处理

    #定义参数
    num_topics = 5
    alpha = 0.1
    beta = 0.01
    max_iter = 100
    documents = pre_text_word
    #训练 LDA 模型
    lda_model = LDA(num_topics, alpha, beta, max_iter)
    lda_model.fit(documents)

    # 获取主题-词分布和文档-主题分布
    topic_word_distribution = [lda_model.calculate_topic_distribution(d, word) for d in range(len(documents)) for word
                               in documents[d]]
    document_topic_distribution = [lda_model.document_topic_counts[d] for d in range(len(documents))]
    top_words_per_topic = lda_model.get_top_words_per_topic(num_top_words=1000)
    # 测试转换函数
    transformed_data = transform_defaultdict(document_topic_distribution)
    result = [extract_mapping(mapping) for mapping in transformed_data]


    # # LDA进行DBI、CH和SC的测试结果，注释掉这一段
    # # 将转换后的LDA特征转换为NumPy数组
    # X = np.array(result)
    # # 使用KMeans进行聚类
    # kmeans = KMeans(n_clusters=num_topics, random_state=42)
    # kmeans.fit(X)
    # labels = kmeans.labels_
    # # 计算聚类评估指标
    # dbi = davies_bouldin_score(X, labels)
    # ch = calinski_harabasz_score(X, labels)
    # sc = silhouette_score(X, labels)
    # # 打印指标结果
    # print("Davies-Bouldin Index:", dbi)
    # print("Calinski-Harabasz Index:", ch)
    # print("Silhouette Coefficient:", sc)

    # 用于存储拼接后的向量
    concatenated_vectors = []

    # 确保 embedding_array 和 result 的长度相同
    # assert len(embedding_array) == len(result)

    # 遍历 embedding_array 和 result 中的每一个元素
    for i in range(len(embedding_array)):
        # 获取当前的768维向量
        embedding_vector = embedding_array[i]
        # 获取当前的5维向量
        additional_vector = np.array(result[i])
        # 将两个向量拼接起来
        concatenated_vector = np.concatenate([embedding_vector, additional_vector])
        # 将拼接后的向量添加到列表中
        concatenated_vectors.append(concatenated_vector)

    # 划分训练集和验证集，80% 用于训练，20% 用于验证
    train_data, valid_data = train_test_split(concatenated_vectors, test_size=0.3, random_state=42)

    # 转换为PyTorch张量
    train_data = torch.tensor(train_data, dtype=torch.float)
    valid_data = torch.tensor(valid_data, dtype=torch.float)
    # 实例化自动编码器
    autoencoder = Autoencoder(input_dim=773, latent_dim=5)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_losses = []
    valid_losses = []

    # 简单的训练循环
    epochs = 30
    for epoch in range(epochs):
        autoencoder.train()
        optimizer.zero_grad()
        decoded, encoded = autoencoder(train_data)
        loss = criterion(decoded, train_data)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 计算验证损失
        autoencoder.eval()
        with torch.no_grad():
            decoded_valid, _ = autoencoder(valid_data)
            valid_loss = criterion(decoded_valid, valid_data)
            valid_losses.append(valid_loss.item())

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item()}, Valid Loss: {valid_loss.item()}")

    # 绘制训练和验证损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # # 转换为PyTorch张量 直接降维使用
    # data = torch.tensor(concatenated_vectors, dtype=torch.float) #拼接向量
    # # data = torch.tensor(embedding_array, dtype=torch.float) #未拼接向量
    # # 实例化自动编码器
    # autoencoder = Autoencoder(input_dim=773, latent_dim=50) #拼接之后的向量长度
    # # autoencoder = Autoencoder(input_dim=768, latent_dim=50) #未拼接之后的向量长度
    # optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()
    #
    # # 简单的训练循环
    # epochs = 30
    # for epoch in range(epochs):
    #     optimizer.zero_grad()
    #     decoded, encoded = autoencoder(data)
    #     loss = criterion(decoded, data)
    #     loss.backward()
    #     optimizer.step()
    #     if epoch % 2 == 0:
    #         print(f"Epoch {epoch}, Loss: {loss.item()}")


    # 使用训练好的自动编码器的编码器部分提取特征
    _, encoded_features = autoencoder(data)

    # 因为KMeans期望的是NumPy数组，所以需要将特征转换回NumPy数组
    encoded_features_np = encoded_features.detach().numpy()

    # 使用KMeans进行聚类
    labels, centers = cluster_features(encoded_features_np, n_clusters=5) # LDA-BERT混合方法
    # labels, centers = cluster_features(result, n_clusters=5) # LDA
    vector_label_pairs = [(encoded_features_np[i], labels[i]) for i in range(len(labels))] # LDA-BERT混合方法
    # vector_label_pairs = [(result[i], labels[i]) for i in range(len(labels))]  # LDA


    # 假设 encoded_features_np 是你通过自动编码器得到的低维特征表示，且labels是对应的聚类标签
    # 转换为适合 scikit-learn 使用的格式
    X = np.array(encoded_features_np)
    labels = np.array(labels)
    # 计算Davies-Bouldin Index
    dbi = davies_bouldin_score(X, labels)
    # 计算Calinski-Harabasz Index
    ch = calinski_harabasz_score(X, labels)
    # 计算Silhouette Coefficient
    # 注意：对于大型数据集，计算SC可能非常耗时，你可以通过设置sample_size参数来计算一部分样本的SC
    sc = silhouette_score(X, labels, sample_size=1000, random_state=42)  # 根据需要调整sample_size和random_state

    # 打印指标结果
    print("Davies-Bouldin Index:", dbi)
    print("Calinski-Harabasz Index:", ch)
    print("Silhouette Coefficient:", sc)

    for dict_, (_, label) in zip(text_label_dict, vector_label_pairs):
        dict_["label"] = f"label_{label}"
    # print(text_label_dict)
    # 输出聚类标签
    print("聚类标签：")
    print(labels)

    # 输出聚类中心
    print("\n聚类中心：")
    for i, center in enumerate(centers):
        print(f"聚类 {i} 的中心: {center}")

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # 计算每个聚类的数据点数和百分比
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_percentages = {label: count / len(labels) for label, count in zip(unique_labels, counts)}

    # 输出聚类标签和百分比
    print("聚类标签及其百分比：")
    for label, percentage in cluster_percentages.items():
        print(f"Cluster {label}: {percentage:.2%}")
    # 使用PCA将编码后的特征降到2维
    pca = PCA(n_components=2)
    encoded_features_2d = pca.fit_transform(encoded_features_np) # LDA-BERT混合方法
    # encoded_features_2d = pca.fit_transform(result) # LDA
    centers_2d = pca.transform(centers)
    # 创建一个散点图，按聚类标签着色
    plt.figure(figsize=(10, 7))
    for i in range(len(centers)):
        # 绘制每个聚类的数据点
        plt.scatter(encoded_features_2d[labels == i, 0], encoded_features_2d[labels == i, 1],
                    label=f'Cluster {i} ({cluster_percentages[i]:.2%})')
        # 绘制聚类中心
        plt.scatter(centers_2d[i, 0], centers_2d[i, 1], c='black', s=200, alpha=0.5)

    plt.title('Cluster visualization with PCA-reduced features')
    plt.legend()
    plt.show()

    # 用于将原始文本打好标签
    # 将text_label_dict转换为DataFrame
    df = pd.DataFrame(text_label_dict)

    # 指定保存Excel文件的路径
    excel_path = 'D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\updated_texts_labels_1.xlsx'

    # 保存DataFrame到Excel文件，不保存索引
    df.to_excel(excel_path, index=False)

    print('文本和标签已成功保存到Excel文件。')




