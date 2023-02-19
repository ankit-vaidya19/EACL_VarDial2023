import torch
from torch.utils.data import DataLoader
import pandas as pd


class VarDialDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, label_mappings):
        super().__init__()
        self.df_path = df_path
        self.label_mappings = label_mappings
        self.df = pd.read_csv(
            self.df_path, sep="\t", names=["Text", "Language"], encoding="latin-1"
        )
        self.df["Label"] = self.df.Language.map(self.label_mappings)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df["Text"].to_list()
        label = self.df["Label"].to_list()
        return text[idx], label[idx]

    def create_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True)

    def weights(self, batch):
        count = self.df["Label"].value_counts().to_list()
        weight_per_class = [0.0] * 3
        N = float(sum(count))
        for i in range(3):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(batch)
        for idx, val in enumerate(batch):
            weight[idx] = weight_per_class[val[1]]
        return torch.utils.data.sampler.WeightedRandomSampler(weight, len(batch))
