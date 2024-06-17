import pandas as pd
from preprocess import *

def excel_to_dict(excel_path):
    """
    读取一个Excel文件，将指定列转换为字典。
    Parameters:
    - excel_path: str, Excel文件的路径。
    Returns:
    - mapping_dict: dict, 包含BADP名称和context&problem的映射字典。
    """
    # 尝试读取Excel文件
    try:
        df = pd.read_excel(excel_path)
        # 创建映射字典，将'BADP名称'列作为键，'context&problem'列作为值
        mapping_dict = df.set_index('BADP名称')['context&problem'].to_dict()
        return mapping_dict
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def preprocess_text_values(mapping_dict):
    """
    对字典中的每个文本值进行预处理。

    Parameters:
    - mapping_dict: dict, 包含文本值的字典。

    Returns:
    - preprocessed_dict: dict, 预处理后的文本值的字典。
    """
    preprocessed_dict = {}
    for key, value in mapping_dict.items():
        # 对每个文本值进行句子级预处理
        processed_sentences = preprocess_sent([value])
        # 对每个文本值进行词级预处理
        processed_words = preprocess_word(processed_sentences)
        # 将预处理后的结果存入新的字典
        preprocessed_dict[key] = processed_words

    return preprocessed_dict


def add_feature_columns_header(excel_path, keys_list):
    """
    添加所有K值作为新列的标题，并将新列插入到指定位置。

    Parameters:
    - excel_path: str, Excel文件的路径。
    - keys_list: list, 包含所有K值名称的列表。

    Returns:
    - None, 但会修改原始Excel文件。
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path)
    # 获取现有的列数
    existing_columns_count = len(df.columns)
    # 计算新列插入的位置
    insert_position = existing_columns_count  # 例如，如果要在末尾添加列，则使用现有列的数量
    # 为每个K值添加一个新列，初始值设为0
    for key in keys_list:
        df.insert(loc=insert_position, column=key, value=0)
        insert_position += 1  # 更新插入位置以便下一个列能够插入在后一个位置
    # 将修改后的DataFrame保存回Excel文件
    df.to_excel(excel_path, index=False)


def update_feature_columns_by_index(excel_path, result_dict):
    """
    根据索引位置更新Excel列，将匹配的单词设为1，不匹配的设为0。

    Parameters:
    - excel_path: str, Excel文件的路径。
    - result_dict: dict, 包含K值到单词数组的映射。

    Returns:
    - None, 但会修改原始Excel文件。
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path)
    # 获取第二列的数据，除了第一行的列标题
    words_column = df.iloc[:, 1]  # 假设第二列的索引是1，第一行（列标题）索引是0

    # 对于result_dict中的每个K值及其关联的单词数组
    for key, words_list in result_dict.items():
        # 展平V中的单词数组，确保没有非迭代对象
        flat_list = []
        for sublist in words_list:
            if isinstance(sublist, list):  # 确保sublist是列表
                flat_list.extend(sublist)  # 展平列表

        # 更新对应的列
        df[key] = words_column.apply(lambda x: 1 if x in flat_list else 0)
    # 将修改后的DataFrame保存回Excel文件
    df.to_excel(excel_path, index=False)





if __name__ == '__main__':
    excel_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\data_BADP_76.xlsx"  # 替换为你的Excel文件路径
    result_dict = excel_to_dict(excel_path)
    if result_dict is not None:
        print(result_dict)

    result = preprocess_text_values(result_dict)
    print(result)

    excel_path_weight = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\得到的数据\\feature_weights_LB.xlsx"  # 替换为你的Excel文件路径
    # 假设excel_path已经设置为您的文件路径
    keys_list = list(result_dict.keys())  # 从result_dict获取所有K值

    # 执行更新操作
    add_feature_columns_header(excel_path_weight, keys_list)
    update_feature_columns_by_index(excel_path_weight, result)