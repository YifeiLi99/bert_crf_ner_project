import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import label2id_path, processed_data_dir
import os

#把bio变成bert可用格式
class CluenerDataset(Dataset):
    def __init__(self, data_file, label2id_file, tokenizer_name, max_length=128):
        self.data_file = data_file
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        # 读标签->id的文件，然后保存进类
        with open(label2id_file, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        # 读取数据
        self.sentences, self.labels = self.load_data()

    #从bio里读取数据，保存为sentences和labels
    def load_data(self):
        #存储全部的词，标签
        sentences, labels = [], []
        #临时存储某句中的字和标签
        words, tags = [], []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                #遇到空白行，断行，保存句子
                if line.strip() == '':
                    if words:
                        sentences.append(words)
                        labels.append(tags)
                        #print(words, tags)
                        words, tags = [], []
                    continue
                #非空白行，存入word和tags，单字单字累成句子
                word, tag = line.strip().split()
                words.append(word)
                tags.append(tag)
            #文件末尾没空行了，别漏，正常加
            if words:
                sentences.append(words)
                labels.append(tags)
        #返回sentences---句子集合，返回labels---标签集合
        return sentences, labels

    def __len__(self):
        #统计有多少个句子
        return len(self.sentences)

    #句子转换逻辑
    def __getitem__(self, idx):
        #输入n，得到第n个句子，第n个标签集合
        words = self.sentences[idx]
        tags = self.labels[idx]

        # BERT分词
        tokens, label_ids = [], []
        #加开始标记
        tokens.append('[CLS]')
        label_ids.append(self.label2id['O'])  # [CLS]默认O类

        #逐个词分BERT子词，并对齐标签
        #！！！兼容性写法：兼容逐字标注或逐词标注数据集，适配子词拆分
        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            #bert分不了的词打未知标签
            if not word_tokens:
                word_tokens = ['[UNK]']
            #把分完子词的句子完整收集起来，送给BERT的token列表
            tokens.extend(word_tokens)

            if len(word_tokens) == 1:
                #只有一个子词时，直接继承原标签
                label_ids.append(self.label2id.get(tag, self.label2id['O']))
            else:
                #有多个子词时，首子词用原标签，后续子词用O
                label_ids.append(self.label2id.get(tag, self.label2id['O']))
                label_ids.extend([self.label2id['O']] * (len(word_tokens) - 1))

        #加结束标记
        tokens.append('[SEP]')
        label_ids.append(self.label2id['O'])  # [SEP]默认O类

        # 把句子通过词表变为数字形式
        #['[CLS]', '清', '华', '大', '学', '[SEP]'] → [101, 1921, 1957, 1920, 2110, 102]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #告诉BERT哪些位置是有效token，哪些是padding
        attention_mask = [1] * len(input_ids)

        # 计算需要多少padding，需要每个句子都一样长变成矩阵
        # 128-现有长度，剩下的后面填0
        padding_length = self.max_length - len(input_ids)
        #不足最大长度，补padding
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            label_ids += [self.label2id['O']] * padding_length
        #长度太长，截断
        else:
            # 优化截断策略：保留 [CLS] 和 [SEP]，截断中间部分
            input_ids = input_ids[:self.max_length - 1] + [self.tokenizer.convert_tokens_to_ids('[SEP]')]
            attention_mask = attention_mask[:self.max_length - 1] + [1]
            label_ids = label_ids[:self.max_length - 1] + [self.label2id['O']]

        #输出。token的索引，那些是有效token，标签
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }

#把多个样本拼成一个样本
#已经padding了，这里单纯打包
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


if __name__ == "__main__":
    # 示例用法
    dataset = CluenerDataset(
        data_file=os.path.join(processed_data_dir, "train.txt"),
        label2id_file=label2id_path,
        tokenizer_name="bert-base-chinese"
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch)
        break
