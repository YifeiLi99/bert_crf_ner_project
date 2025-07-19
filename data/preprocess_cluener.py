import os
import json
from collections import Counter
from config import raw_data_dir,processed_data_dir

# 如果预处理目录不存在，则创建
os.makedirs(processed_data_dir, exist_ok=True)

# 定义BIO标注生成函数，将原始标注字典转化为BIO标签列表
def get_bio_labels(text, label_dict):
    labels = ['O'] * len(text)  # 初始化标签为O（非实体）
    for entity_type, entities in label_dict.items():  # 遍历每种实体类别
        for entity_text, positions in entities.items():  # 遍历每个实体及其出现的位置
            for start, end in positions:  # 对每个位置进行标注
                labels[start] = f"B-{entity_type}"  # 实体起始位置用B标注
                for i in range(start + 1, end + 1):
                    labels[i] = f"I-{entity_type}"  # 实体内部用I标注
    return labels


# 处理单个文件，将原始json格式数据转为BIO标注的txt格式
def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fr, \
            open(output_file, 'w', encoding='utf-8') as fw:
        for line in fr:
            sample = json.loads(line.strip())  # 每行一个json对象
            text = sample["text"]  # 获取句子文本
            label = sample.get("label", {})  # 获取标注信息，可能没有（如测试集）
            bio_labels = get_bio_labels(text, label)  # 生成BIO标签

            # 确保标签长度与文本长度一致
            assert len(text) == len(bio_labels), f"Length mismatch at line: {line}"

            # 按每个字+标签写入文件
            for char, tag in zip(text, bio_labels):
                fw.write(f"{char} {tag}\n")
            fw.write("\n")  # 每句话之间空一行


# 根据训练和验证集的BIO文件生成label2id.json
def build_label2id(output_file, files):
    label_counter = Counter()  # 用于统计所有出现过的标签
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == "":  # 空行跳过
                    continue
                _, tag = line.strip().split()  # 取出标签
                label_counter[tag] += 1

    labels = sorted(label_counter.keys())  # 标签按字母排序，保持稳定性
    label2id = {label: idx for idx, label in enumerate(labels)}  # 分配ID

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print(f"label2id saved to {output_file}")


if __name__ == "__main__":
    # 定义输入输出文件路径
    input_train = os.path.join(raw_data_dir, "train.json")
    input_dev = os.path.join(raw_data_dir, "dev.json")
    output_train = os.path.join(processed_data_dir, "train.txt")
    output_dev = os.path.join(processed_data_dir, "dev.txt")

    # 处理训练集和验证集
    print("Processing train.json...")
    process_file(input_train, output_train)
    print("Processing dev.json...")
    process_file(input_dev, output_dev)

    # 生成标签到id的映射文件
    build_label2id(os.path.join(processed_data_dir, "label2id.json"), [output_train, output_dev])
    print("Preprocessing finished!")
