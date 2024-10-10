import re
from string import ascii_lowercase
from collections import defaultdict
from pyctcdecode import build_ctcdecoder

import torch
import numpy as np


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, use_lm=True, kenlm_model_path="3-gram.arpa", **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        self.use_lm = use_lm
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        with open("librispeech-vocab.txt") as f:
                unigrams = [t.lower() for t in f.read().strip().split("\n")]
        self.decoder  = build_ctcdecoder([self.EMPTY_TOK] + [i.upper() for i in self.alphabet], kenlm_model_path=kenlm_model_path, unigrams=unigrams)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
    
        collapsed_inds = []
        prev = None
        for ind in inds:
            ind = int(ind)
            if ind != prev:
                collapsed_inds.append(ind)
                prev = ind

        empty_i = self.char2ind[self.EMPTY_TOK]
        return "".join([self.ind2char[int(ind)] for ind in collapsed_inds if int(ind) != empty_i]).strip()

    def ctc_beam_search(self, log_probs: torch.tensor, beam_size: int):
        if self.use_lm:
            if isinstance(log_probs, torch.Tensor):
                log_probs = log_probs.detach().cpu().numpy()
            return self.decoder.decode(log_probs, beam_size).lower()
        dp = {
            ('', self.EMPTY_TOK): 1.0
        }
        
        def extend_path_and_merge(dp, next_token_probs: torch.tensor, ind2char: dict):
            new_dp = defaultdict(float)
            for ind, next_token_prob in enumerate(next_token_probs.T):
                cur_char = ind2char[ind]
                for (prefix, last_char), v in dp.items():
                    if cur_char == last_char:
                        new_prefix = prefix
                    else:
                        if cur_char != self.EMPTY_TOK:
                            new_prefix = prefix + cur_char
                        else:
                            new_prefix = prefix
                    new_dp[(new_prefix, cur_char)] += v + next_token_prob
            return new_dp

        def truncate_paths(dp, beam_size):
            return dict(sorted(list(dp.items()), key=lambda x: -x[1], reverse=True)[ : beam_size])
        
        print(log_probs.shape)
        for probs in log_probs:
            dp = extend_path_and_merge(dp=dp, next_token_probs=probs, ind2char=self.ind2char)
            dp = truncate_paths(dp, beam_size)

        dp = [(prefix, proba) for (prefix, _), proba in dp.items()]

        return dp[0][0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
