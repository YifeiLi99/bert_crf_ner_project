import os
import torch

# 获取项目根目录
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 数据相关路径
data_dir = os.path.join(BASE_DIR, "data")
# 定义原始数据和预处理后数据的路径
raw_data_dir = os.path.join(data_dir, "raw", "cluener_public")
processed_data_dir = os.path.join(data_dir, "processed", "cluener_public")

# 模型权重路径
weights_dir = os.path.join(BASE_DIR, "weights")

# 日志路径
logs_dir = os.path.join(BASE_DIR, "logs")

# 推理接口路径
gradio_app_path = os.path.join(BASE_DIR, "inference_service", "gradio_app.py")
fastapi_service_path = os.path.join(BASE_DIR, "inference_service", "fastapi_service.py")

#设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练超参数
num_epochs = 20
#64-128为最佳区间
batch_size = 64
learning_rate = 5e-5
#最大语句长度
max_length=128
#bert预训练模型名
pretrained_model_name = 'hfl/chinese-roberta-wwm-ext'
#KL 损失默认加权系数
kl_weight = 5.0

# 标签文件
label2id_path = os.path.join(processed_data_dir, "label2id.json")

if __name__ == "__main__":
    print("项目根目录:", BASE_DIR)
    print("原始数据路径:", raw_data_dir)
    print("预处理数据路径:", processed_data_dir)
    print("模型权重路径:", weights_dir)
    print("日志路径:", logs_dir)
    print("训练轮数:", num_epochs, "批次大小:", batch_size, "学习率:", learning_rate)
