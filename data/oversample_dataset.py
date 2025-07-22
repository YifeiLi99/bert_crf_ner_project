from collections import Counter
import os
from config import processed_data_dir

def read_sentences(train_file):
    sentences = []
    sentence = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                sentence.append(line)
        if sentence:
            sentences.append(sentence)
    return sentences


def count_label_distribution(train_file, small_labels=None):
    """
    统计训练集中各类别标签的出现次数

    :param train_file: 训练集文件路径
    :param small_labels: 可选，是否只统计小类别的分布
    :return: Counter 字典
    """
    label_counter = Counter()
    sentences = read_sentences(train_file)

    for sentence in sentences:
        labels = [line.split()[-1] for line in sentence]
        if small_labels:
            labels = [label for label in labels if label in small_labels]
        label_counter.update(labels)

    return label_counter


def oversample_dataset(train_file, output_file, small_labels, small_multiplier=4, normal_multiplier=1):
    """
    根据小类别标签对训练集进行过采样

    :param train_file: 原始训练集路径
    :param output_file: 过采样后输出文件路径
    :param small_labels: 小类别标签列表
    :param small_multiplier: 小类别样本重复倍数
    :param normal_multiplier: 普通样本重复倍数
    """
    sentences = read_sentences(train_file)
    new_sentences = []

    for sentence in sentences:
        labels = [line.split()[-1] for line in sentence]
        repeat = small_multiplier if any(label in small_labels for label in labels) else normal_multiplier
        new_sentences.extend([sentence] * repeat)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in new_sentences:
            for line in sentence:
                f.write(f"{line}\n")  # 正确的格式化字符串
            f.write("\n")  # 空行分隔句子
    print(f"已生成过采样训练集：{output_file}，总句子数：{len(new_sentences)}")


if __name__ == "__main__":
    train_file = os.path.join(processed_data_dir, "train.txt")
    output_file = os.path.join(processed_data_dir, "train_oversample.txt")

    small_labels = ['B-book', 'I-book', 'B-scene', 'I-scene', 'B-address', 'I-address']

    label_counter = count_label_distribution(train_file)
    print("所有类别分布:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

    small_counter = count_label_distribution(train_file, small_labels=small_labels)
    print("\n小类别分布:")
    for label, count in small_counter.items():
        print(f"{label}: {count}")

    oversample_dataset(train_file, output_file, small_labels, small_multiplier=4, normal_multiplier=1)
