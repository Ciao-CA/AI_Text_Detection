# AI_Text_Detection

model: roberta-large

显卡:
> NVIDIA GeForce RTX 3090 * 5(实验室的)

训练集:
> MEGA

模型
> Roberta-large

数据集
> MEGA: Multilingual Evaluation of Generative AI\
> id: 2303.12528

### 下载

```
git clone https://github.com/Ciao-CA/AI_Text_Detection.git
conda create -n ai-detection python=3.9.18
conda activate ai-detection
pip install -r AI_Text_Detection/requirements
unzip AI_Text_Detection/model/roberta-large.zip -d AI_Text_Detection/model/roberta-large
unzip AI_Text_Detection/winner_model/robert-large_text_classifier.zip -d AI_Text_Detection/winner_model/robert-large_text_classifier
```

### 训练

> CUDA_VISIBLE_DEVICES='0,5,7,8,9' python homework/train_eval.py

### 测试

> CUDA_VISIBLE_DEVICES='0,5,7,8,9' python homework/test.py
