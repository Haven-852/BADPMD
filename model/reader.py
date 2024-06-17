import openpyxl
import pandas as pd


def read_text_column_from_excel(file_path, column_name):
    """
    从Excel文件中读取指定标题的一列文本集
    :param file_path: Excel文件路径
    :param column_name: 指定的列标题
    :return: 文本集列表
    """
    # 使用pandas库读取Excel文件
    df = pd.read_excel(file_path)

    # 获取指定列的文本集
    text_column = df[column_name].tolist()

    return text_column

