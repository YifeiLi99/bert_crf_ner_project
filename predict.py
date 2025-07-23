import torch
from transformers import BertTokenizer
from model.bert_crf_model import BertCRFModel
from config import label2id_path, weights_dir, pretrained_model_name, DEVICE
import json
import os

# ---------------------- 参数 ----------------------
MODEL_WEIGHTS = os.path.join(weights_dir, 'best_model004_final.pt')

# ---------------------- 加载 label2id 和 id2label ----------------------
with open(label2id_path, 'r', encoding='utf-8') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# ---------------------- 加载分词器与模型 ----------------------
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
model = BertCRFModel(label2id, pretrained_model_name=pretrained_model_name)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# ---------------------- 推理函数 ----------------------
def predict_sentence(sentence):
    words = list(sentence)
    tokens = ['[CLS]'] + words + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)

    # padding
    max_length = 128
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

    input_ids = torch.tensor([input_ids]).to(DEVICE)
    attention_mask = torch.tensor([attention_mask]).to(DEVICE)

    with torch.no_grad():
        preds = model(input_ids, attention_mask)[0]  # batch里只有一条

    # 去除[CLS]和[SEP]
    predicted_labels = preds[1:len(words)+1]
    predicted_tags = [id2label[label_id] for label_id in predicted_labels]

    return list(zip(words, predicted_tags))


if __name__ == "__main__":
    while True:
        sentence = input("请输入一段中文文本：")
        result = predict_sentence(sentence)
        print("预测结果：")
        for word, tag in result:
            print(f"{word}\t{tag}")
