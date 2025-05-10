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

'''
git clone 
'''
训练
> CUDA_VISIBLE_DEVICES='0,5,7,8,9' python homework/train_eval.py

测试
> CUDA_VISIBLE_DEVICES='0,5,7,8,9' python homework/test.py
