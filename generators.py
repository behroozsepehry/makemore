from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Tuple
import random
import math


class NameGenerator(ABC):
    def __init__(self, alphabet: List[str], end_token: str):
        self.alphabet: List[str] = alphabet
        self.end_token: str = end_token
        self.vocab_size: int = len(alphabet)

    @abstractmethod
    def predict(self, batch: List[List[str]]) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def train(self, batch: List[str]) -> None:
        pass

    @abstractmethod
    def get_generator_name(self) -> str:
        pass

    def generate(self) -> str:
        name: List[str] = []
        while True:
            probs: Dict[str, float] = self.predict([name])[0]
            next_char: str = random.choices(
                list(probs.keys()), weights=list(probs.values())
            )[0]
            if next_char == self.end_token:
                break
            name.append(next_char)
        return "".join(name)

    def generate_names(self, num_names: int = 10) -> List[str]:
        return [self.generate() for _ in range(num_names)]

    def evaluate_batch(self, batch: List[str]) -> float:
        total_log_likelihood: float = 0
        total_tokens: int = 0

        for sequence in batch:
            sequence_log_likelihood: float = 0
            for i in range(len(sequence)):
                prefix: str = sequence[:i]
                probs: Dict[str, float] = self.predict([prefix])[0]

                char: str = sequence[i]
                if char in probs:
                    sequence_log_likelihood += math.log(probs[char])
                else:
                    sequence_log_likelihood += math.log(1e-10)
                total_tokens += 1

            total_log_likelihood += sequence_log_likelihood

        average_log_likelihood: float = (
            total_log_likelihood / total_tokens if total_tokens > 0 else float("-inf")
        )
        return average_log_likelihood


class DumbNameGenerator(NameGenerator):
    def predict(self, batch: List[List[str]]) -> List[Dict[str, float]]:
        return [{char: 1.0 / self.vocab_size for char in self.alphabet} for _ in batch]

    def train(self, batch: List[str]) -> None:
        pass

    def get_generator_name(self) -> str:
        return "Dumb"


class NGramNameGenerator(NameGenerator):
    def __init__(self, alphabet: List[str], end_token: str, k: int):
        super().__init__(alphabet, end_token)
        self.k: int = k
        self.ngram_counts: Dict[str, Dict[str, int]] = {}
        self.context_counts: Dict[str, int] = {}

    def train(self, batch: List[str]) -> None:
        for name in batch:
            padded_name: str = "<" * self.k + name
            for i in range(len(padded_name) - self.k):
                context: str = padded_name[i : i + self.k]
                next_char: str = padded_name[i + self.k]

                if context not in self.ngram_counts:
                    self.ngram_counts[context] = {}

                if next_char not in self.ngram_counts[context]:
                    self.ngram_counts[context][next_char] = 0

                if context not in self.context_counts:
                    self.context_counts[context] = 0

                self.ngram_counts[context][next_char] += 1
                self.context_counts[context] += 1

    def predict(self, batch: List[List[str]]) -> List[Dict[str, float]]:
        predictions: List[Dict[str, float]] = []

        for sequence in batch:
            padded_sequence: str = "<" * self.k + "".join(sequence)
            context: str = padded_sequence[-self.k :]

            if context in self.ngram_counts:
                total_count: int = self.context_counts[context]
                probs: Dict[str, float] = {
                    char: count / total_count
                    for char, count in self.ngram_counts[context].items()
                }

                for char in self.alphabet:
                    if char not in probs:
                        probs[char] = 1e-10

                norm_factor: float = sum(probs.values())
                probs = {char: prob / norm_factor for char, prob in probs.items()}
            else:
                probs = {char: 1.0 / self.vocab_size for char in self.alphabet}

            predictions.append(probs)

        return predictions

    def get_generator_name(self) -> str:
        return f"N-Gram (k={self.k})"
