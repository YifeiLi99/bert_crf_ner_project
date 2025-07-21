import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import label2id_path, processed_data_dir
import os

class CluenerDataset(Dataset):
    def __init__(self, data_file, label2id_file, tokenizer_name, max_length=128):
        self.data_file = data_file
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        # 加载标签字典
        with open(label2id_file, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)

        # 读取数据
        self.sentences, self.labels = self.load_data()

    def load_data(self):
        sentences, labels = [], []
        words, tags = [], []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '':
                    if words:
                        sentences.append(words)
                        labels.append(tags)
                        words, tags = [], []
                    continue
                word, tag = line.strip().split()
                words.append(word)
                tags.append(tag)
            if words:
                sentences.append(words)
                labels.append(tags)
        return sentences, labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        # BERT分词
        tokens, label_ids = [], []
        tokens.append('[CLS]')
        label_ids.append(self.label2id['O'])  # [CLS]默认O类

        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = ['[UNK]']
            tokens.extend(word_tokens)
            # 第一个subword用原标签，后续用O
            label_ids.append(self.label2id.get(tag, self.label2id['O']))
            label_ids.extend([self.label2id['O']] * (len(word_tokens) - 1))

        tokens.append('[SEP]')
        label_ids.append(self.label2id['O'])  # [SEP]默认O类

        # 转换成BERT输入ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            label_ids += [self.label2id['O']] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            label_ids = label_ids[:self.max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }


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
