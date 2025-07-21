import os
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from data.dataset_cluener import CluenerDataset, collate_fn
from model.bert_crf_model import BertCRFModel
from config import processed_data_dir, label2id_path, weights_dir, logs_dir, num_epochs, batch_size, learning_rate, \
    max_length, pretrained_model_name, DEVICE
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------------------- 参数 ----------------------
CHECKPOINT_PATH = os.path.join(weights_dir, 'best_model.pt')
LOG_PATH = os.path.join(logs_dir, 'train_log.txt')
TENSORBOARD_LOG_DIR = os.path.join(logs_dir, 'tensorboard')
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# ---------------------- 日志 ----------------------
tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
log_file = open(LOG_PATH, 'w', encoding='utf-8')

# ---------------------- 加载标签 ----------------------
with open(label2id_path, 'r', encoding='utf-8') as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# ---------------------- 数据集 ----------------------
train_file = os.path.join(processed_data_dir, 'train.txt')
val_file = os.path.join(processed_data_dir, 'dev.txt')
train_dataset = CluenerDataset(train_file, label2id_path, pretrained_model_name, max_length)
val_dataset = CluenerDataset(val_file, label2id_path, pretrained_model_name, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------------- 模型 ----------------------
model = BertCRFModel(label2id, pretrained_model_name=pretrained_model_name)
model.to(DEVICE)

# ---------------------- 优化器 & Scheduler ----------------------
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * num_epochs
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# ---------------------- EarlyStopping ----------------------
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        else:
            self.counter += 1
            return False  # no improvement

    def should_stop(self):
        return self.counter >= self.patience


early_stopper = EarlyStopping(patience=3)


# ---------------------- 评估函数 ----------------------
def evaluate():
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
            attention_mask = torch.tensor(batch['attention_mask']).to(DEVICE)
            labels = torch.tensor(batch['labels']).to(DEVICE)

            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            preds = model(input_ids, attention_mask)  # List[List[int]]
            preds = [p[:labels.shape[1]] for p in preds]
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(val_loader)

    # 扁平化用于f1_score
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)

    f1 = f1_score(flat_labels, flat_preds, average='macro', zero_division=0)

    report = classification_report(flat_labels, flat_preds, target_names=list(label2id.keys()), zero_division=0)

    print(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(flat_labels, flat_preds)
    print(f"Confusion Matrix:\n{cm}")

    return avg_loss, f1


if __name__ == "__main__":
    print("\n========== Start Training ==========")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}")
        for batch in loop:
            input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
            attention_mask = torch.tensor(batch['attention_mask']).to(DEVICE)
            labels = torch.tensor(batch['labels']).to(DEVICE)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_f1 = evaluate()

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val F1 = {val_f1:.4f}")
        tb_writer.add_scalars("Loss", {'train': avg_train_loss, 'val': val_loss}, epoch)
        tb_writer.add_scalar("Val_F1", val_f1, epoch)
        log_file.write(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}\n")
        log_file.flush()

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"\n✅ New best model saved to {CHECKPOINT_PATH}")
            best_val_loss = val_loss

        if early_stopper.step(val_loss):
            print(f"✅ Validation Loss improved to {val_loss:.4f}")
        else:
            print(f"❗ No improvement for {early_stopper.counter} epoch(s)")

        if early_stopper.should_stop():
            print("⛔ Early stopping triggered")
            break

    print(f"\n✅ Best validation loss: {best_val_loss:.4f}")
    print("✅ Training finished")
    tb_writer.close()
    log_file.close()
