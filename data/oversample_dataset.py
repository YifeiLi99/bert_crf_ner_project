from collections import Counter
import os
from config import processed_data_dir


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


def read_sentences(train_file):
    """
    按空行分句，返回句子列表
    """
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


def get_sampling_rules(label_counter):
    """
    根据类别统计次数动态生成采样倍率字典
    """
    sampling_rules = {}
    for label, count in label_counter.items():
        if count > 10000:
            sampling_rules[label] = 1
        elif count > 5000:
            sampling_rules[label] = 2
        elif count > 1000:
            sampling_rules[label] = 3
        else:
            sampling_rules[label] = 4
    print("\n类别采样倍率:")
    for label, mult in sampling_rules.items():
        print(f"{label}: 重复 {mult} 倍")
    return sampling_rules


def oversample_dataset(train_file, output_file, sampling_rules):
    """
    根据类别动态采样规则对训练集进行过采样
    """
    sentences = read_sentences(train_file)
    new_sentences = []

    for sentence in sentences:
        labels = [line.split()[-1] for line in sentence]
        repeat = max([sampling_rules.get(label, 1) for label in labels])
        new_sentences.extend([sentence] * repeat)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in new_sentences:
            for line in sentence:
                f.write(f"{line}\n")
            f.write("\n")
    print(f"已生成过采样训练集：{output_file}，总句子数：{len(new_sentences)}")


if __name__ == "__main__":
    train_file = os.path.join(processed_data_dir, "train.txt")
    output_file = os.path.join(processed_data_dir, "train_oversample.txt")

    label_counter = count_label_distribution(train_file)
    print("所有类别分布:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

    sampling_rules = get_sampling_rules(label_counter)

    oversample_dataset(train_file, output_file, sampling_rules)
