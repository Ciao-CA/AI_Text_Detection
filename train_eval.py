import torch
from torch.utils.data import Dataset, DataLoader
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe.iloc[index, 0]
        label = self.dataframe.iloc[index, 1]

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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0

    loop = tqdm(data_loader, desc="Training", leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        if isinstance(loss, torch.Tensor) and len(loss.shape) > 0:
            loss = loss.mean()
        losses.append(loss.item())

        # 计算预测准确率
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_postfix(loss=loss.item())

    return np.mean(losses), correct_predictions.double() / total_predictions

def evaluate_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    true_labels = []

    loop = tqdm(data_loader, desc="Evaluating", leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

    return predictions, true_labels


def save_model(model, model_path):
    if torch.cuda.device_count() > 1:
        model.module.save_pretrained(model_path)
    else:
        model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)  # 自动将输入分发到多个 GPU 上
    model = model.to(device)
    return model


def train():

    print("train")
    print(model_name)
    print(model_save_path)
    
    train_data_f = pd.read_csv('AI_Text_Detection/data/MAGA/train.csv')
    val_data_f = pd.read_csv('AI_Text_Detection/data/MAGA/valid.csv')

    train_size = int(len(train_data_f))
    val_size  = int(len(val_data_f))

    train_df, val_df = train_data_f[:train_size], val_data_f[:val_size]

    train_dataset = TextDataset(train_df, tokenizer, max_len)
    val_dataset = TextDataset(val_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(model_name, device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_model(model, train_loader, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")

        save_model(model, f"{model_save_path}{epoch}")

        predictions, true_labels = evaluate_model(model, val_loader, device)
        print(classification_report(true_labels, predictions))
        

if __name__ == "__main__":
    model_name = 'AI_Text_Detection/model/roberta-large'
    model_save_path = 'AI_Text_Detection/model/robert-large_text_classifier_'
    max_len = 512  # Roberta-large最长514
    batch_size = 32
    epochs = 2
    learning_rate = 2e-6

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train()
