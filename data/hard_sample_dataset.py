import torch
from transformers import BertTokenizer
from tqdm import tqdm
from model.bert_crf_model import BertCRFModel  # 按你实际的项目路径调整
from config import processed_data_dir, weights_dir, pretrained_model_name, label2id_path, DEVICE
import os
import json

# ---------------------- 加载标签 ----------------------
with open(label2id_path, 'r', encoding='utf-8') as f:
    label2id = json.load(f)

def extract_hard_samples(train_file, output_file, model_path, pretrained_model_name, max_length=128):
    """
    用训练好的模型推理 train.txt，输出错分句子
    """
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    model = BertCRFModel(label2id, pretrained_model_name=pretrained_model_name)
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    id2label = {v: k for k, v in label2id.items()}

    def parse_sentence(lines):
        tokens, labels = [], []
        for line in lines:
            word, label = line.strip().split()
            tokens.append(word)
            labels.append(label)
        return tokens, labels

    hard_sentences = []

    with open(train_file, 'r', encoding='utf-8') as f:
        sentence_lines = []
        for line in tqdm(f):
            line = line.strip()
            if line == "":
                if sentence_lines:
                    tokens, true_labels = parse_sentence(sentence_lines)
                    encodings = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length, return_tensors="pt")
                    input_ids = encodings["input_ids"].to(DEVICE)
                    attention_mask = encodings["attention_mask"].to(DEVICE)

                    with torch.no_grad():
                        predictions = model(input_ids, attention_mask)[0]  # list[int]

                    pred_labels = [id2label[idx] for idx in predictions]

                    # 裁剪到tokens长度（不算CLS、SEP）
                    valid_len = len(tokens)
                    pred_labels = pred_labels[:valid_len]

                    # 判断整句是否有错
                    if pred_labels != true_labels:
                        hard_sentences.append("\n".join(sentence_lines) + "\n")

                    sentence_lines = []
            else:
                sentence_lines.append(line)

    print(f"错分句子数：{len(hard_sentences)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in hard_sentences:
            f.write(sent + "\n")
    print(f"已保存 hard_samples 至：{output_file}")


if __name__ == "__main__":
    model_path = os.path.join(weights_dir, 'best_model004_oversample.pt')  # 按需替换

    train_file = os.path.join(processed_data_dir, "train.txt")
    output_file = os.path.join(processed_data_dir, "hard_samples.txt")

    extract_hard_samples(train_file, output_file, model_path, pretrained_model_name)
