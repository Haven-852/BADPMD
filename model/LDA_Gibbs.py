import numpy as np
import random
from collections import defaultdict
import heapq


class LDA:
    def __init__(self, num_topics, alpha, beta, max_iter):
        self.num_topics = num_topics
        self.alpha = alpha  # 文档-主题分布的超参数
        self.beta = beta    # 主题-词分布的超参数
        self.max_iter = max_iter

    def fit(self, documents):
        self.documents = documents
        self.vocab = list(set([word for document in documents for word in document]))
        self.vocab_size = len(self.vocab)

        # 初始化主题分布
        self.topic_assignments = [[random.randint(0, self.num_topics - 1) for _ in range(len(document))] for document in documents]

        # 初始化计数器
        self.document_topic_counts = [defaultdict(int) for _ in range(len(documents))]
        self.topic_word_counts = [defaultdict(int) for _ in range(self.num_topics)]
        self.topic_counts = [0 for _ in range(self.num_topics)]
        self.document_lengths = [len(document) for document in documents]

        # 随机初始化单词的主题
        for d in range(len(documents)):
            for i in range(len(documents[d])):
                word = documents[d][i]
                topic = self.topic_assignments[d][i]
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.topic_counts[topic] += 1

        # Gibbs Sampling 迭代
        for _ in range(self.max_iter):
            for d in range(len(documents)):
                for i in range(len(documents[d])):
                    word = documents[d][i]
                    topic = self.topic_assignments[d][i]

                    # 从当前主题中移除单词计数
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1

                    # 计算新的主题分布
                    topic_distribution = self.calculate_topic_distribution(d, word)
                    new_topic = np.random.choice(self.num_topics, p=topic_distribution)

                    # 更新主题分配
                    self.topic_assignments[d][i] = new_topic

                    # 更新计数器
                    self.document_topic_counts[d][new_topic] += 1
                    self.topic_word_counts[new_topic][word] += 1
                    self.topic_counts[new_topic] += 1

    def calculate_topic_distribution(self, d, word):
        alpha_sum = self.num_topics * self.alpha
        beta_sum = self.vocab_size * self.beta

        topic_distribution = [
            ((self.topic_word_counts[k][word] + self.beta) / (self.topic_counts[k] + beta_sum)) *
            ((self.document_topic_counts[d][k] + self.alpha) / (self.document_lengths[d] + alpha_sum))
            for k in range(self.num_topics)
        ]

        topic_distribution /= np.sum(topic_distribution)
        return topic_distribution

    def get_top_words_per_topic(self, num_top_words=1000):
        top_words_per_topic = []

        for topic in range(self.num_topics):
            word_probabilities = [
                (word, (self.topic_word_counts[topic][word] + self.beta) / (
                            self.topic_counts[topic] + self.vocab_size * self.beta))
                for word in self.vocab
            ]
            top_words = heapq.nlargest(num_top_words, word_probabilities, key=lambda x: x[1])
            top_words_per_topic.append(top_words)

        return top_words_per_topic

# 转换函数
def transform_defaultdict(defaultdict_list):
    result_list = []
    for d in defaultdict_list:
        sorted_keys = sorted(d.keys())
        result_str = ','.join([f"{key}: {d[key]}" for key in sorted_keys])
        result_list.append(result_str)
    return result_list

def extract_mapping(mapping):
    values = [0] * 5 # 创建一个包含5个0的数组
    pairs = mapping.split(",")  # 按逗号分割映射
    for pair in pairs:
        key, value = pair.split(":")  # 按冒号分割键值对
        key = int(key.strip())  # 去除空格并将键转换为整数
        value = int(value.strip())  # 去除空格并将值转换为整数
        values[key] = value  # 将值放入对应的位置
    return values


