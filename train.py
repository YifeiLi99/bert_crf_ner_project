import os
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from data.dataset_cluener import CluenerDataset, collate_fn
from model.bert_crf_model import BertCRFModel
from config import processed_data_dir, label2id_path, weights_dir, logs_dir, num_epochs, batch_size, learning_rate, \
    max_length, pretrained_model_name, DEVICE, kl_weight, warmup
import json
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- 参数 ----------------------
CHECKPOINT_PATH = os.path.join(weights_dir, 'best_model004_final.pt')
LOG_PATH = os.path.join(logs_dir, 'train_log004_final.txt')
TENSORBOARD_LOG_DIR = os.path.join(logs_dir, 'tensorboard004_final')
cm_save_path = os.path.join(logs_dir, 'confusion_matrix004_final')
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(cm_save_path, exist_ok=True)

# ---------------------- 日志 ----------------------
tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
log_file = open(LOG_PATH, 'w', encoding='utf-8')

# ---------------------- 加载标签 ----------------------
with open(label2id_path, 'r', encoding='utf-8') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# ---------------------- 数据集 ----------------------
train_file = os.path.join(processed_data_dir, 'hard_samples.txt')
val_file = os.path.join(processed_data_dir, 'dev.txt')
train_dataset = CluenerDataset(train_file, label2id_path, pretrained_model_name, max_length)
val_dataset = CluenerDataset(val_file, label2id_path, pretrained_model_name, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

# ---------------------- 模型 ----------------------
model = BertCRFModel(label2id, pretrained_model_name=pretrained_model_name, kl_weight=kl_weight)
model.to(DEVICE)

#读取训练好的模型继续训练
#不用就给注销掉
#model.load_state_dict(torch.load(os.path.join(weights_dir, "best_model004_oversample.pt"), map_location=DEVICE))
#print("成功加载，正在进行进行hard samples finetune训练")

# ---------------------- 优化器 & Scheduler ----------------------
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    #BERT+CRF组合极容易初期震荡，0.2-0.4的预热比较正常
    num_warmup_steps=int(warmup * total_steps),
    num_training_steps=total_steps
)


# ---------------------- EarlyStopping ----------------------
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_f1 = 0
        self.counter = 0

    def step(self, val_f1):
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            return False  # no improvement

    def should_stop(self):
        return self.counter >= self.patience

#配合20的epoch，5-7的耐心值
early_stopper = EarlyStopping(patience=1)


# ---------------------- 评估函数（改造版） ----------------------
def evaluate():
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
            attention_mask = torch.tensor(batch['attention_mask']).to(DEVICE)
            labels = torch.tensor(batch['labels']).to(DEVICE)

            # ===== 验证时只单次forward，不用R-Drop逻辑 =====
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = model.dropout(outputs.last_hidden_state)
            emissions = model.classifier(sequence_output)

            loss = -model.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            total_loss += loss.item()

            preds = model.crf.decode(emissions, mask=attention_mask.bool())

            for pred_seq, label_seq, mask_seq in zip(preds, labels.cpu().tolist(), attention_mask.cpu().tolist()):
                valid_len = sum(mask_seq)
                all_preds.extend(pred_seq[:valid_len])
                all_labels.extend(label_seq[:valid_len])

    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    report = classification_report(all_labels, all_preds, target_names=list(label2id.keys()), zero_division=0)
    print(f"\nClassification Report (Val F1 = {f1:.4f}):\n{report}")

    # 混淆矩阵绘图
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(label2id.keys()), yticklabels=list(label2id.keys()), ax=ax)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title(f'Confusion Matrix (F1={f1:.4f})')

    # 保存图片
    cm_save_picture = os.path.join(cm_save_path, f"confusion_matrix_epoch_{epoch + 1}.png")
    plt.tight_layout()
    plt.savefig(cm_save_picture)
    plt.close()

    print(f"混淆矩阵图已保存: {cm_save_picture}")

    # TensorBoard也可同步记录混淆矩阵图片（可选）
    if tb_writer is not None:
        import torchvision
        cm_img = torchvision.transforms.ToTensor()(plt.imread(cm_save_picture))
        tb_writer.add_image(f"Confusion_Matrix/Epoch_{epoch+1}", cm_img, epoch)

    return avg_loss, f1



# --------------------- 主程序 -------------------------
if __name__ == "__main__":
    print("\n================== 训练开始 ==================")
    best_val_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        #初始化计数器
        total_train_loss = 0
        total_crf_loss = 0
        total_kl_loss = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}")
        for batch in loop:
            input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
            attention_mask = torch.tensor(batch['attention_mask']).to(DEVICE)
            labels = torch.tensor(batch['labels']).to(DEVICE)

            total_loss, crf_loss, kl_loss = model(input_ids, attention_mask, labels)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += total_loss.item()
            total_crf_loss += crf_loss.item()
            total_kl_loss += kl_loss.item()

            loop.set_postfix(total=total_loss.item(), crf=crf_loss.item(), kl=kl_loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_crf_loss = total_crf_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        val_loss, val_f1 = evaluate()

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}")
        tb_writer.add_scalars("Loss", {'train': avg_train_loss, 'train_crf': avg_crf_loss, 'train_kl': avg_kl_loss,
                                       'val': val_loss}, epoch)
        tb_writer.add_scalar("Val_F1", val_f1, epoch)
        log_file.write(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, CRF Loss={avg_crf_loss:.4f}, KL Loss={avg_kl_loss:.4f}\n")
        log_file.write(f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}\n\n")
        log_file.flush()

        if val_f1 > best_val_f1:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"\n最佳模型已保存至{CHECKPOINT_PATH}")
            best_val_f1 = val_f1

        if early_stopper.step(val_f1):
            print(f"val f1 提升至{val_f1:.4f}")
        else:
            print(f"注意！已有{early_stopper.counter}个epoch无提升")

        if early_stopper.should_stop():
            print("启动早停")
            break

    print(f"\n最佳val f1: {best_val_f1:.4f}")
    print("训练完成")
    tb_writer.close()
    log_file.close()
