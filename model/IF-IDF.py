from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from preprocess import *
from reader import *



def save_to_excel(words, weights):
    # Create a DataFrame with ID, Word, and Weight columns
    df = pd.DataFrame({
        'ID': range(1, len(words) + 1),
        'Word': words,
        'Weight': weights
    })

    # Define the file path
    file_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\feature_weight_TF_IDF.xlsx"

    # Save the DataFrame to an Excel file
    df.to_excel(file_path, index=False)
    print(f'Data saved to {file_path}')

if __name__ == '__main__':
    excel_file = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\updated_texts_labels.xlsx"  # Excel文件路径
    column_title = "text"  # 指定的列标题
    text_data = read_text_column_from_excel(excel_file, column_title)
    pre_text_data = preprocess_sent(text_data)  # 文本级处理
    pre_text_word = preprocess_word(pre_text_data)  # 词级处理
    # 将每个文档的单词列表连接为一个字符串
    documents = [" ".join(doc) for doc in pre_text_word]

    # 将所有文档合并为一个字符串，这里假设所有文档组成一个大的文档
    combined_document = " ".join(documents)

    # 初始化TF-IDF向量化器，并计算单词的TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vector = tfidf_vectorizer.fit_transform([combined_document])

    # 将得到的TF-IDF向量转换为数组
    tfidf_array = tfidf_vector.toarray().flatten()

    # 获取所有单词
    words = tfidf_vectorizer.get_feature_names_out()

    # 创建DataFrame来存储单词和它们对应的TF-IDF值
    df = pd.DataFrame({'Word': words, 'TF-IDF': tfidf_array})

    # 将单词按照TF-IDF值降序排列
    df = df.sort_values(by='TF-IDF', ascending=False)

    # 显示单词和它们的TF-IDF值
    df = df.reset_index(drop=True)
    print(df)

    # 如果想得到单词和它们TF-IDF值的列表
    words_list = df['Word'].tolist()
    tfidf_list = df['TF-IDF'].tolist()

    # 显示向量空间模型，第一行是单词，第二行是权值
    print("Vector Space Model:")
    print(words_list)  # 单词
    print(tfidf_list)  # 权值
    save_to_excel(words_list, tfidf_list)
