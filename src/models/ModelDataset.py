import torch
import pandas as pd

# from datasets import Dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def read_and_split(dataset_path: str, shuffle: bool = False):
    df = pd.read_csv(dataset_path)
    if shuffle:
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        df = df.sample(frac=1).reset_index(drop=True)
    return df["text"].tolist(), df["benign"].tolist()


class ModelDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            dataset_path: str = None,
            cache_load_from_path: str = None,
            cache_save_to_path: str = None,
            shuffle: bool = False,
    ):
        self.tokenizer = tokenizer
        self.X_data, self.labels = read_and_split(dataset_path, shuffle)
        self.encodings = []
        self.cache_load_from_path = cache_load_from_path
        self.cache_save_to_path = cache_save_to_path
        print("Read data and labels from", dataset_path)

        # read from cache if file exists
        try:
            self.encodings = torch.load(cache_load_from_path)
            print("Data loaded from cache")
        except FileNotFoundError:
            print("No cache file with path", cache_load_from_path)
            self.encode_data()
            self.cache_data_if_needed()

    def encode_data(self):
        self.encodings = self.tokenizer(
            self.X_data, truncation=True, padding=True, max_length=400)
        print("Data encoded.")

    def cache_data_if_needed(self):
        if self.cache_save_to_path:
            torch.save(self.encodings, self.cache_save_to_path)
            print("Data cached to", self.cache_save_to_path)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

