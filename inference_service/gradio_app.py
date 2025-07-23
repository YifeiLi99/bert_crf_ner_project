import gradio as gr
import torch
from transformers import BertTokenizerFast
from model.bert_crf_model import BertCRFModel
from config import label2id_path, pretrained_model_name, weights_dir, DEVICE
import os
import json

# ---------------------- 加载label2id ----------------------
with open(label2id_path, 'r', encoding='utf-8') as f:
    label2id = json.load(f)

# id2label字典，用于从id恢复到标签名
id2label = {v: k for k, v in label2id.items()}

# ---------------------- 加载tokenizer和模型 ----------------------
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
model = BertCRFModel(label2id, pretrained_model_name=pretrained_model_name).to(DEVICE)
model_path = os.path.join(weights_dir, "best_model004_final.pt")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

def ner_predict(text):
    """
    NER推理函数：输入一段文本，返回高亮实体字符串
    """
    if not text.strip():
        return "⚠️ 输入不能为空"

    tokens = list(text.strip())
    encodings = tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)
    offset_mapping = encodings["offset_mapping"][0].tolist()

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)[0]

    pred_labels = []
    token_idx = 0

    for offset, pred_id in zip(offset_mapping, predictions):
        if offset[0] == 0 and offset[1] != 0:
            if token_idx < len(tokens):
                label = id2label.get(pred_id, "O")
                pred_labels.append(label)
                token_idx += 1

    if len(pred_labels) != len(tokens):
        return f"⚠️ 标签数量与输入字符数不符！ tokens={len(tokens)}, preds={len(pred_labels)}"

    # ---------------------- 实体高亮展示 ----------------------
    result = ""
    current_entity = ""
    current_label = ""

    for char, label in zip(tokens, pred_labels):
        if label.startswith("B-"):
            if current_entity:
                result += f"<mark style='background-color: #ffd966'>{current_entity} [{current_label}]</mark>"
                current_entity = ""
            current_entity = char
            current_label = label[2:]
        elif label.startswith("I-") and current_entity:
            current_entity += char
        else:
            if current_entity:
                result += f"<mark style='background-color: #ffd966'>{current_entity} [{current_label}]</mark>"
                current_entity = ""
            result += char

    if current_entity:
        result += f"<mark style='background-color: #ffd966'>{current_entity} [{current_label}]</mark>"

    return result

def gradio_interface(text):
    """
    Gradio接口封装函数，返回HTML高亮结果
    """
    try:
        return ner_predict(text)
    except Exception as e:
        return f"❌ 推理异常: {str(e)}"

if __name__ == "__main__":
    demo = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(lines=4, placeholder="请输入一段中文文本"),
        outputs=gr.HTML(),
        title="CLUENER 中文命名实体识别",
        description="输入中文句子，输出高亮标注的NER实体",
        examples=[
            "小明毕业于清华大学电机系，现在在上海工作。",
            "我想买一本三体。",
            "北京市海淀区的中关村软件园。"
        ]
    )
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False)
