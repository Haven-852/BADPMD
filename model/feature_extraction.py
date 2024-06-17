from preprocess import *
import numpy as np
import pandas as pd
from LDA_Gibbs import *
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from sklearn.feature_extraction.text import TfidfVectorizer


def loadData(datapath):
    data = pd.read_excel(datapath, header=0)
    dataX = data.iloc[:, 0].values  # 第一列为文本数据
    labels = data.iloc[:, 1].values  # 第二列为文本标签

    # 将标签类别用 0, 1, 2 ,3 ,4表示
    labels[np.where(labels == "label_0")] = 0
    labels[np.where(labels == "label_1")] = 1
    labels[np.where(labels == "label_2")] = 2
    labels[np.where(labels == "label_3")] = 3
    labels[np.where(labels == "label_4")] = 4
    return dataX, labels

# 扩充主题数量与文本单词数量一致
def expand_topic_labels(labels, pre_text_word):
    expanded_labels = []
    # 遍历原始主题数组和预处理后的文本
    for label, words in zip(labels, pre_text_word):
        # 扩充主题标签，次数等于当前文档的单词数量
        expanded_labels.extend([label] * len(words))
    return expanded_labels

# 定义提取特征词的函数
def filter_feature_words(topic_word_distribution, expanded_labels, pre_text_words):
    # 初始化5个主题的特征词列表
    feature_words = [[] for _ in range(5)]

    # 遍历每个单词的主题分布向量、扩充后的主题标签和预处理后的单词
    for dist, label, word in zip(topic_word_distribution, expanded_labels, pre_text_words):
        # 检查这个单词的主题分布是否符合它所属的主题标签
        if np.argmax(dist) == label:
            # 如果符合，加入到相应主题的特征词列表中
            feature_words[label].append(word)

    return feature_words

def create_documents_from_feature_words(feature_words):
    # 为每个类别创建一个文档字符串
    # feature_words 是一个嵌套列表，其中包含每个类别的单词列表
    documents = [" ".join([" ".join(sublist) if isinstance(sublist, list) else sublist for sublist in word_list]) for
                 word_list in feature_words]
    return documents

def compute_tfidf_vectors(documents):
    # 第二步: 使用TF-IDF转换文档到向量空间模型
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
    return tfidf_vectorizer, tfidf_vectors

def create_vector_space_model(tfidf_vectorizer, tfidf_vectors):
    # 第三步: 创建向量空间模型，第一行为单词，第二行为权值
    feature_names = tfidf_vectorizer.get_feature_names_out()
    weights = tfidf_vectors.toarray()
    vector_space_model_df = pd.DataFrame(weights, columns=feature_names)
    return vector_space_model_df

def print_vector_space_models(vector_space_model_df):
    # 打印向量空间模型的内容
    if isinstance(vector_space_model_df, pd.DataFrame):
        for index, row in vector_space_model_df.iterrows():
            print(f"Vector Space Model for Document {index+1}:")
            print(row)
            print("\n")
    else:
        print("Provided argument is not a pandas DataFrame.")

def create_single_document_from_feature_words(feature_words):
    # 遍历feature_words中的每个列表，确保每个子列表被平铺，并且所有元素都是字符串
    flat_list = [str(item) for sublist in feature_words for item in sublist]
    # 将平铺后的列表连接成一个单一的字符串文档
    document = " ".join(flat_list)
    return document

if __name__ == '__main__':
    # 读取有标签的文本
    read_text_labels = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\updated_texts_labels.xlsx" # 测试文件名text_label 文件名updated_texts_labels
    dataX, labels = loadData(read_text_labels)  # 读取数据
    # 文本预处理
    pre_text_data = preprocess_sent(dataX) # 文本级处理
    pre_text_word = preprocess_word(pre_text_data) # 词级处理
    #定义参数
    num_topics = 5
    alpha = 0.1
    beta = 0.01
    max_iter = 100
    documents = pre_text_word

    dictionary = Dictionary(documents)
    #训练 LDA 模型
    corpus = [dictionary.doc2bow(text) for text in pre_text_word]

    lda_model = LDA(num_topics, alpha, beta, max_iter)
    lda_model.fit(documents)

    # 获取主题-词分布和文档-主题分布
    topic_word_distribution = [lda_model.calculate_topic_distribution(d, word) for d in range(len(documents)) for word
                               in documents[d]]
    expanded_labels = expand_topic_labels(labels, pre_text_word)
    feature_words = filter_feature_words(topic_word_distribution, expanded_labels, pre_text_word)

    # 将特征词列表转换为文档
    document = create_single_document_from_feature_words(feature_words)

    # 把这个单一的文档字符串放入一个列表中
    documents = [document]

    # 计算TF-IDF向量
    tfidf_vectorizer, tfidf_vectors = compute_tfidf_vectors(documents)

    # 创建向量空间模型
    vector_space_model_df = create_vector_space_model(tfidf_vectorizer, tfidf_vectors)

    # 按TF-IDF权重降序排列特征词
    feature_weights = vector_space_model_df.sum().sort_values(ascending=False)

    # 输出排序后的特征词及其权重
    print(feature_weights)

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(list(feature_weights.items()), columns=['Word', 'Weight'])

    # Sort the DataFrame by 'Weight' in descending order (if not already sorted)
    df = df.sort_values(by='Weight', ascending=False)

    # Reset the index to get an 'ID' column starting from 1
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    df['ID'] = df.index

    # Reorder the DataFrame columns
    df = df[['ID', 'Word', 'Weight']]

    # Specify your Excel file path
    excel_file_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\feature_weights.xlsx"

    # Write the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)

    print(f"Data has been successfully saved to {excel_file_path}")
