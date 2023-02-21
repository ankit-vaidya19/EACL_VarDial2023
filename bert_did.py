import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from tqdm import *
from sklearn.metrics import classification_report
from data import VarDialDataset


class DialectID(nn.Module):
    def __init__(self, model_name, label_mappings, lang):
        super().__init__()
        self.model_name = model_name
        self.label_mappings = label_mappings
        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.transformer = AutoModel.from_pretrained(self.model_name)
        self.lin = nn.Linear(768, 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.push_to_device()

    def forward(self, batch):
        x = self.transformer(**batch).last_hidden_state[:, 0, :]
        x = self.lin(x)
        return x

    def push_to_device(self):
        self.transformer.to(self.device)
        self.lin.to(self.device)

    def calculate_f1(self, predictions, labels):
        return classification_report(
            torch.concat(labels, dim=0).cpu(),
            torch.concat(predictions, dim=0).cpu(),
            digits=4,
        )

    def fit(self, train_loader, val_loader):
        optim = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-6)
        criterion = nn.CrossEntropyLoss()
        self.push_to_device()
        best_acc = 0
        for epoch in range(20):
            train_loss = []
            val_loss = []
            train_preds = []
            val_preds = []
            train_labels = []
            val_labels = []
            num_correct = 0
            num_samples = 0
            print(f"Epoch - {epoch+1}/20")
            self.train()
            for batch in tqdm(train_loader):
                batch[0] = self.tokenizer(
                    text=list(batch[0]),
                    return_attention_mask=True,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                text = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = batch[1].to(self.device)
                outputs = self(text)
                loss = criterion(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss)
                train_preds.append(outputs.argmax(dim=-1))
                train_labels.append(batch[1])
            print(f"Train loss - {sum(train_loss)/len(train_loss)}")
            print(self.calculate_f1(train_preds, train_labels))
            self.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    batch[0] = self.tokenizer(
                        text=list(batch[0]),
                        return_attention_mask=True,
                        max_length=256,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    text = {k: v.to(self.device) for k, v in batch[0].items()}
                    labels = batch[1].to(self.device)
                    outputs = self(text)
                    loss = criterion(outputs, labels)
                    val_loss.append(loss)
                    val_preds.append(outputs.argmax(dim=-1))
                    val_labels.append(batch[1])
                    _, predictions = outputs.max(1)
                    num_correct += (predictions == labels).sum()
                    num_samples += predictions.size(0)
                val_acc = (float(num_correct) / float(num_samples)) * 100
                print(f"Validation loss - {sum(val_loss)/len(val_loss)}")
                print(self.calculate_f1(val_preds, val_labels))
                if val_acc > best_acc:
                    best_acc = val_acc
                    print("Saved")
                    torch.save(self.state_dict(), f"./{self.lang}-bert-ws-{epoch}.pt")
