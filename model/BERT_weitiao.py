from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
import BERT
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

def map_design_pattern_to_label(pattern_name):
    # 设计模式列表
    design_patterns = [
        "Encrypting On-Chain Data", "Tokenisation", "Blockchain Anchor", "Payment Channel(State Channel)",
        "Snapshotting", "Node Sync", "Establish Genesis", "Hard Fork", "State Initialisation", "Exchange Transfer",
        "Virtual Machine Emulation", "Smart Contract Translation", "Measure migration quality", "Develop with Production Data",
        "State Aggregation", "Migrate along Domain", "Partitions", "Daily Quality Reports", "Event log", "Limit storage",
        "Mapping vs Array", "Fixed size", "Default value", "Minimize on-chain data", "Emergency Stop",
        "Multiple Authorisation/Multisignature", "Off-Chain Secret Enabled Dynamic Authorisation", "Embedded Permission",
        "Hot and Cold Wallet Storage", "Key Shards", "Time-Constrained Access", "One-Off Access", "X-Confirmation",
        "Security Deposit", "Master and Sub Key Generation", "Identifier Registry", "Multiple Registration",
        "Blockchain and Social Media Account Pair", "Dual Resolution", "Delegate List", "Selective Content Generation",
        "Token Registry", "Policy Contract", "Token Burning", "Escrow", "Seller Credential", "Stealth Address",
        "Token Swap", "Authorised Spender", "Ownership", "Oracle", "Decentralised Oracle", "Reverse Oracle", "Voting",
        "Legal and Smart Contract Pair", "Token Template", "Contract Registry", "Data Contract/Data Segregation",
        "Diamond(Multi-Facet Proxy)", "Factory Contract", "Incentive Execution", "Proxy(Contract Relay)",
        "Decentralised Applications (DApps)", "Semi Decentralised Applications (Semi-DApp)"
    ]

    # 创建设计模式到标签的映射
    pattern_to_label = {pattern: i+1 for i, pattern in enumerate(design_patterns)}

    # 获取输入设计模式的标签
    return pattern_to_label.get(pattern_name, "Unknown pattern")

def get_labels_from_names(design_pattern_names):
    return [map_design_pattern_to_label(name) for name in design_pattern_names]

def preprocess_data(texts, labels, max_len=64, batch_size=32):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()

    # 编码标签
    encoded_labels = label_encoder.fit_transform(labels)

    # Tokenize 文本
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # 转为Tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(encoded_labels)

    # 创建 DataLoader
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader, label_encoder


from transformers import BertForSequenceClassification, AdamW


def initialize_model(num_labels):
    # 加载预训练模型
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )

    # 优化器设置
    optimizer = AdamW(model.parameters(), lr=2e-5)

    return model, optimizer


def train_model(model, dataloader, optimizer, device, epochs=4):
    for epoch_i in range(epochs):
        print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))
        total_loss = 0
        model.train()

        for step, batch in enumerate(dataloader):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        avg_train_loss = total_loss / len(dataloader)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("Training complete!")


def predict(model, tokenizer, label_encoder, text, device):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(ids, token_type_ids=None, attention_mask=mask)
        logits = outputs[0]

    predicted_class_id = logits.argmax().item()
    predicted_class = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_class

if __name__ == '__main__':
    excel_file = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\data_DPP-review.xlsx"  # Excel文件路径
    column_title = "raw_data"  # 指定的列标题
    column_title_label = "BADP"
    text_data = read_text_column_from_excel(excel_file, column_title)
    label = read_text_column_from_excel(excel_file, column_title_label)
    # labels = get_labels_from_names(label)
    # 过滤并处理可以被BERT处理的文本
    successful_texts = []
    successful_labels = []  # 用于存储可以被BERT处理的文本对应的标签

    for text, labels in zip(text_data, label):
        if isinstance(text, str) or (isinstance(text, list) and all(isinstance(item, str) for item in text)):
            successful_texts.append(text)
            successful_labels.append(labels)
        else:
            # 如果不是字符串或字符串列表，则跳过这个文本
            continue
    mapped_labels = get_labels_from_names(successful_labels)

    # 假设 successful_texts 和 mapped_labels 是可用的
    texts = successful_texts
    labels = mapped_labels

    # 步骤 1: 预处理数据
    dataloader, label_encoder = preprocess_data(texts, labels)

    # 步骤 2: 初始化模型
    num_labels = len(set(labels))  # 标签的数量
    model, optimizer = initialize_model(num_labels)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 步骤 3: 训练模型
    train_model(model, dataloader, optimizer, device, epochs=4)

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 步骤 4: 进行预测（假设有一些新文本）
    sample_text = "The execution environment of a blockchain is self-contained. It can only access information present in the data and transactions on the blockchain. Smart contracts running on a blockchain are pure functions by design. The state of external systems is not directly accessible to smart contracts. Yet, function calls in smart contracts sometimes need to access state of the external world."
    predicted_label = predict(model, tokenizer, label_encoder, sample_text, device)
    print("Predicted Label:", predicted_label)

