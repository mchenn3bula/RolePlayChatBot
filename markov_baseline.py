# markov_baseline.py

import math
from collections import defaultdict
from datasets import load_from_disk

class MarkovChain:
    def __init__(self, n: int = 2):
        """
        Initialize an n-gram Markov chain model.
        Args:
            n (int): order of the model (uses n-grams for context).
        """
        self.n = n
        self.model = defaultdict(list)

    def train(self, corpus: list[str]):
        """
        Train on an in-memory list of sentences.
        Args:
            corpus (list[str]): List of raw text strings.
        """
        for sentence in corpus:
            tokens = sentence.strip().split()
            if len(tokens) < self.n:
                continue
            for i in range(len(tokens) - self.n):
                prefix = tuple(tokens[i : i + self.n])
                next_token = tokens[i + self.n]
                self.model[prefix].append(next_token)

    def train_dataset(self, dataset_path: str, text_field: str = "text"):
        """
        Train directly from an on-disk Arrow dataset (sharded .arrow files).
        Args:
            dataset_path (str): folder containing data-*.arrow shards
            text_field (str): name of the text column in the dataset
        """
        ds = load_from_disk(dataset_path)
        for ex in ds:
            sentence = ex[text_field]
            tokens = sentence.strip().split()
            if len(tokens) < self.n:
                continue
            for i in range(len(tokens) - self.n):
                prefix = tuple(tokens[i : i + self.n])
                next_token = tokens[i + self.n]
                self.model[prefix].append(next_token)

    def generate(self, seed: list[str], max_tokens: int = 50) -> list[str]:
        """
        Generate a sequence of tokens given an initial seed sequence.
        Args:
            seed (list[str]): starting tokens of length >= n
            max_tokens (int): maximum number of tokens to generate
        Returns:
            list[str]: generated token sequence (including seed)
        """
        if len(seed) < self.n:
            raise ValueError(f"Seed length must be at least {self.n}")
        output = seed.copy()
        for _ in range(max_tokens):
            prefix = tuple(output[-self.n :])
            choices = self.model.get(prefix)
            if not choices:
                break
            next_tok = random.choice(choices)
            output.append(next_tok)
        return output

    def respond_to(self, input_text: str, max_tokens: int = 50) -> str:
        """
        Simple response by seeding on the last n words of input_text.
        Args:
            input_text (str): raw input sentence
        Returns:
            str: generated continuation
        """
        tokens = input_text.strip().split()
        generated = self.generate(tokens, max_tokens)
        return " ".join(generated)

    def perplexity(self, corpus: list[str], unk_prob: float = 1e-6) -> float:
        """
        Compute perplexity on an in-memory list of sentences.
        Args:
            corpus (list[str]): list of raw text strings
            unk_prob (float): fallback probability for unseen n-grams
        Returns:
            float: perplexity
        """
        log_prob_sum = 0.0
        token_count = 0
        prefix_counts = {p: len(sufs) for p, sufs in self.model.items()}

        for sentence in corpus:
            tokens = sentence.strip().split()
            if len(tokens) <= self.n:
                continue
            for i in range(self.n, len(tokens)):
                token_count += 1
                prefix = tuple(tokens[i - self.n : i])
                target = tokens[i]
                sufs = self.model.get(prefix)
                if sufs:
                    cnt = sufs.count(target)
                    prob = cnt / prefix_counts[prefix] if cnt > 0 else unk_prob
                else:
                    prob = unk_prob
                log_prob_sum += math.log(prob)

        if token_count == 0:
            raise ValueError("No valid n-grams found in corpus for perplexity.")

        H = -log_prob_sum / token_count
        return math.exp(H)

    def perplexity_dataset(
        self,
        dataset_path: str,
        text_field: str = "text",
        unk_prob: float = 1e-6
    ) -> float:
        """
        Compute perplexity on an on-disk Arrow dataset.
        Args:
            dataset_path (str): folder with data-*.arrow shards
            text_field (str): name of the text column
            unk_prob (float): fallback probability for unseen n-grams
        Returns:
            float: perplexity
        """
        ds = load_from_disk(dataset_path)

        log_prob_sum = 0.0
        token_count = 0
        prefix_counts = {p: len(sufs) for p, sufs in self.model.items()}

        for ex in ds:
            sentence = ex[text_field]
            tokens = sentence.strip().split()
            if len(tokens) <= self.n:
                continue
            for i in range(self.n, len(tokens)):
                token_count += 1
                prefix = tuple(tokens[i - self.n : i])
                target = tokens[i]
                sufs = self.model.get(prefix)
                if sufs:
                    cnt = sufs.count(target)
                    prob = cnt / prefix_counts[prefix] if cnt > 0 else unk_prob
                else:
                    prob = unk_prob
                log_prob_sum += math.log(prob)

        if token_count == 0:
            raise ValueError("No valid n-grams found in dataset for perplexity.")

        H = -log_prob_sum / token_count
        return math.exp(H)
