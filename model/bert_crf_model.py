import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF  # 当前使用 torchcrf (kmkurn版)


# bert + crf模型
class BertCRFModel(nn.Module):
    def __init__(self, label2id, pretrained_model_name='bert-base-chinese', dropout_prob=0.1):
        super(BertCRFModel, self).__init__()
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.num_labels = len(label2id)

        # bert backbone
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # dropout防止过拟合
        self.dropout = nn.Dropout(dropout_prob)

        # 线性层分类器，将BERT的hidden_size转换为类别数
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        # CRF层，torchcrf支持batch_first=True，无需转置
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT获取上下文特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # 线性层得到每个token的logits
        emissions = self.classifier(sequence_output)  # [batch_size, seq_length, num_labels]

        if labels is not None:
            # 训练模式，返回CRF损失，torchcrf支持直接reduction
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            # 推理模式，返回预测路径（不需要labels）
            prediction = self.crf.decode(emissions, mask=attention_mask.bool())
            return prediction


if __name__ == "__main__":
    from config import label2id_path
    import json

    # 加载label2id字典
    with open(label2id_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)

    # 模型实例化
    model = BertCRFModel(label2id)

    # 构造假输入验证维度是否对得上
    dummy_input = torch.randint(0, 100, (2, 10))  # [batch_size=2, seq_length=10]
    dummy_mask = torch.ones(2, 10).long()
    dummy_labels = torch.randint(0, len(label2id), (2, 10)).long()

    # 前向推理返回loss
    loss = model(dummy_input, dummy_mask, dummy_labels)
    print(f"Dummy Loss: {loss.item()}")

    # 推理返回预测路径
    prediction = model(dummy_input, dummy_mask)
    print(f"Dummy Prediction: {prediction}")
