import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF  # 当前使用 torchcrf (kmkurn版)
import torch.nn.functional as F  # 新增，用于R-Drop KL散度计算


# bert + crf模型
class BertCRFModel(nn.Module):
    def __init__(self, label2id, pretrained_model_name='bert-base-chinese', dropout_prob=0.1, kl_weight=5.0):
        super(BertCRFModel, self).__init__()
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        #计算类别数，用于分类器输出维度
        self.num_labels = len(label2id)
        self.kl_weight = kl_weight  # 新增，R-Drop中KL散度的权重系数

        #调用 Huggingface 的 BertModel 加载预训练BERT
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
        emissions1 = self.classifier(sequence_output)  # [batch_size, seq_length, num_labels]

        if labels is not None:
            # 训练模式，返回CRF损失，torchcrf支持直接reduction
            #crf返回对数似然值，越大越好（负无穷到0）。加负号让整体越小越好（0到正无穷）
            crf_loss = -self.crf(emissions1, labels, mask=attention_mask.bool(), reduction='mean')

            # ========== R-Drop新增逻辑 ========== #
            # 第二次forward，获得第二份emissions
            outputs2 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output2 = self.dropout(outputs2.last_hidden_state)
            emissions2 = self.classifier(sequence_output2)

            # KL散度 loss，计算两个forward输出的分布差异
            log_probs1 = F.log_softmax(emissions1, dim=-1)
            log_probs2 = F.log_softmax(emissions2, dim=-1)
            kl_loss = F.kl_div(log_probs1, log_probs2, log_target=True, reduction='batchmean') + \
                      F.kl_div(log_probs2, log_probs1, log_target=True, reduction='batchmean')
            kl_loss = kl_loss / 2

            # 总loss = 单次CRF loss + KL loss加权
            total_loss = crf_loss + self.kl_weight * kl_loss
            return total_loss, crf_loss.detach(), kl_loss.detach()
        else:
            # 推理模式，返回预测路径（不需要labels）
            # 推理时只forward一次
            prediction = self.crf.decode(emissions1, mask=attention_mask.bool())
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
    total_loss, crf_loss, kl_loss = model(dummy_input, dummy_mask, dummy_labels)
    print(f"Dummy Total Loss: {total_loss.item()}, CRF Loss: {crf_loss.item()}, KL Loss: {kl_loss.item()}")

    # 推理返回预测路径
    prediction = model(dummy_input, dummy_mask)
    print(f"Dummy Prediction: {prediction}")
