import re
from string import ascii_lowercase
from collections import defaultdict
from pyctcdecode import build_ctcdecoder
from pyctcdecode import Alphabet, BeamSearchDecoderCTC

import torch
import numpy as np


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, kenlm_model_path="3-gram.arpa", vocab_path="librispeech-vocab.txt", **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if kenlm_model_path is not None:
            with open(vocab_path) as f:
                unigrams = [line.strip() for line in f.readlines()] 
            self.decoder_lm = build_ctcdecoder(labels=[""] + self.alphabet, kenlm_model_path=kenlm_model_path, unigrams=unigrams)
        self.decoder_no_lm = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

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
        res = []
        empty_i = self.char2ind[self.EMPTY_TOK]
        last = empty_i
        for i in inds:
            if i != last and i != empty_i:
                res.append(self.ind2char[i])
                continue
            last = i
        return "".join(res)


    def ctc_beam_search(self, use_lm: bool, log_probs: torch.tensor, probs: torch.tensor, logits: torch.tensor, beam_size: int):
        if use_lm:
            if isinstance(logits, torch.Tensor):
                logits = logits.detach().cpu().numpy()
            return self.decoder_lm.decode(logits, 100).lower()
        
        dp = {("", self.EMPTY_TOK): 1.0}

        def extend_path_and_merge(dp, next_token_probs: np.array, ind2char: dict):
            new_dp = defaultdict(float)
            for ind, next_token_prob in enumerate(next_token_probs):
                cur_char = ind2char[ind]
                for (prefix, last_char), v in dp.items():
                    if cur_char == last_char:
                        new_prefix = prefix
                    else:
                        if cur_char != self.EMPTY_TOK:
                            new_prefix = prefix + cur_char
                        else:
                            new_prefix = prefix
                    new_dp[(new_prefix, cur_char)] += (v * next_token_prob) 
            return new_dp

        def truncate_paths(dp, beam_size):
            d = dict(sorted(list(dp.items()), key = lambda x: x[1], reverse = True)[:beam_size])
            return d
        
        for i in np.exp(log_probs):
            dp = extend_path_and_merge(dp, i, self.ind2char)
            dp = truncate_paths(dp, beam_size)

        dp = [(prefix, proba) for (prefix, _), proba in dp.items()]

        return dp[0][0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
