
# BERT-CRF 中文命名实体识别（CLUENER2020）

本项目基于BERT-CRF模型实现中文命名实体识别任务，使用CLUENER2020公开数据集，完整复现从数据预处理到推理部署的AI工程项目，适合NLP入门与简历项目展示。

## 📌 项目亮点
- 完整AI工程化流程：数据清洗、模型训练、推理、API部署
- 支持Gradio可视化界面和FastAPI在线服务
- BIO标注规范，模块化代码架构，易于迁移复用
- 可拓展至其他中文NER任务

## 📁 目录结构
```
bert_crf_ner_project/
├── config.py
├── data/
├── model/
├── train.py
├── predict.py
├── inference_service/
├── logs/
├── weights/
└── README.md
```

## 🚀 快速开始

### 1. 环境准备
```bash
conda create -n nerenv python=3.9
conda activate nerenv
pip install -r requirements.txt
```

### 2. 数据准备
将CLUENER2020数据集放置于 `data/raw/cluener2020/`。

### 3. 预处理
```bash
python data/preprocess_cluener.py
```

### 4. 训练模型
```bash
python train.py
```

### 5. 推理/部署
```bash
# 单句预测
python predict.py

# 启动 Gradio Demo
python inference_service/gradio_app.py
```

## 📊 数据集
- [CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020)

## 🏆 效果指标
| Metric | Dev Set |
|---------|----------|
| F1-score | xx.xx% |

## 💡 TODO
- [ ] 融合轻量化BERT（如DistilBERT）
- [ ] 多任务NER
- [ ] 医疗/金融NER迁移

---

## 📜 License
本项目仅用于学习交流，禁止商业用途。
