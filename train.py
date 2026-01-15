import pandas as pd
import argparse
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from utils import clean_text
from torch.nn.utils.rnn import pad_sequence


# LOAD DATA
def load_data(path):
    df = pd.read_json(path, lines=True)

    df["headline"] = df["headline"].fillna("")
    df["short_description"] = df["short_description"].fillna("")
    df["category"] = df["category"].fillna("")

    df["text"] = df["headline"] + " " + df["short_description"]

    df = df.dropna(subset=["text", "category"])
    df = df[["text", "category"]]
    df.columns = ["text", "label"]

    return df


# DATASET CLASS
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder):
        self.texts = texts
        self.labels = label_encoder.transform(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def encode_text(self, text):
        tokens = text.split()
        ids = [self.tokenizer.get(t, 1) for t in tokens]  # 1 = <UNK>
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        text = self.encode_text(self.texts[idx])
        label = int(self.labels[idx])
        return text, label


def collate_batch(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True)
    return padded.long(), torch.tensor(labels, dtype=torch.long)


# LSTM MODEL
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        out = self.fc(h[-1])
        return out


# MAIN FUNCTION
def main(args):
    print("📥 Loading data...")
    df = load_data(args.input)

    print("🧹 Cleaning text...")
    df["clean"] = df["text"].apply(clean_text)

    print("🔡 Building vocabulary...")
    words = set()
    for t in df["clean"]:
        words.update(t.split())

    tokenizer = {w: i + 2 for i, w in enumerate(words)}
    tokenizer["<PAD>"] = 0
    tokenizer["<UNK>"] = 1

    print(f"🔠 Vocab size = {len(tokenizer)}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"])
    num_classes = len(label_encoder.classes_)

    train_df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    train_set = NewsDataset(train_df["clean"], train_df["label"], tokenizer, label_encoder)
    test_set = NewsDataset(test_df["clean"], test_df["label"], tokenizer, label_encoder)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_batch)

    model = LSTMClassifier(
        vocab_size=len(tokenizer),
        emb_dim=128,
        hidden_dim=256,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training LSTM...")
    for epoch in range(5):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X = X.long()
            y = y.long()

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss = {total_loss:.4f}")

    print("📊 Evaluating...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.long()
            y = y.long()

            out = model(X)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("F1 Macro:", f1_score(all_labels, all_preds, average="macro"))

    print("💾 Saving artifacts...")
    os.makedirs("artifacts", exist_ok=True)

    torch.save(model.state_dict(), "artifacts/model_lstm.pth")
    pickle.dump(tokenizer, open("artifacts/tokenizer.pkl", "wb"))
    pickle.dump(label_encoder, open("artifacts/labels.pkl", "wb"))

    print("✔ Done!")


# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="News_Category_Dataset_v3.json")
    args = parser.parse_args()
    main(args)
