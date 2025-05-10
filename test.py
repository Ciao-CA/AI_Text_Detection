import torch
from torch.utils.data import Dataset, DataLoader
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        prompt = str(self.dataframe.iloc[index, 0])
        text = str(self.dataframe.iloc[index, 1])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'prompt': prompt,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def evaluate_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    probabilities = []

    loop = tqdm(data_loader, desc="Evaluating", leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)


        probability = torch.sigmoid(logits)  # 形状 [3, 1]

        predictions.extend(preds.cpu().tolist())
        probabilities.extend(probability.cpu().tolist())
        # texts.extend(text)
        # prompts.extend(prompt)
    probabilities = [ item[1] for item in probabilities]
        
    return predictions, probabilities


def load_model(model_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # 自动将输入分发到多个 GPU 上
    model = model.to(device)
    return model
        

def eval():

    print("eval")
    print(model_path)
    model = load_model(model_path, device)

    # test_df = pd.read_excel('AI_Text_Detection/data/UCAS_AISAD_TEXT-test1.xlsx')
    test_df = pd.read_csv('AI_Text_Detection/data/UCAS_AISAD_TEXT-test1.csv')
    test_dataset = TextDataset(test_df, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions, probabilities = evaluate_model(model, test_loader, device)
    

    df = pd.DataFrame({
        'label': predictions
    })
    df.to_csv('AI_Text_Detection/data/result/predictions.csv', index=False)

    df = pd.DataFrame({
        'label': probabilities
    })
    df.to_csv('AI_Text_Detection/data/result/probabilities.csv', index=False)

    # print(f"predictions: {predictions}")


if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间

    max_len = 512  # Roberta-large最长514
    batch_size = 1024
    epochs = 2
    learning_rate = 2e-6

    model_name = '/netcache/huggingface/roberta-large'

    model_path = 'AI_Text_Detection/winner_model/robert-large_text_classifier'

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval()

    end_time = time.time()  # 记录开始时间
    print(f"运行时间：{end_time - start_time}")

    # CUDA_VISIBLE_DEVICES='0,5,7,8,9' python homework/test.py
