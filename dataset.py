import requests
import os
from typing import List, Generator


class NameDataset:
    def __init__(self, names_file: str, url: str):
        self.names_file: str = names_file
        self.url: str = url
        self.end_token: str = ">"
        self.download_if_not_exists()
        self.names: List[str] = self.load_names()
        self.alphabet: List[str] = list(set("".join(self.names) + self.end_token))

    def download_if_not_exists(self) -> None:
        data_dir = os.path.dirname(self.names_file)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.exists(self.names_file):
            print(f"Downloading {self.names_file}...")
            response: requests.Response = requests.get(self.url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(self.names_file, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print(f"{self.names_file} already exists. Skipping download.")

    def load_names(self) -> List[str]:
        with open(self.names_file, "r") as f:
            return [name.strip().lower() + self.end_token for name in f]

    def get_names(self) -> List[str]:
        return self.names

    def get_alphabet(self) -> List[str]:
        return self.alphabet

    def get_end_token(self) -> str:
        return self.end_token

    def get_batch(
        self, batch_size: int = 32, shuffle: bool = True
    ) -> Generator[List[str], None, None]:
        import random

        indices: List[int] = list(range(len(self.names)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            yield [
                self.names[indices[j]]
                for j in range(i, min(i + batch_size, len(indices)))
            ]
