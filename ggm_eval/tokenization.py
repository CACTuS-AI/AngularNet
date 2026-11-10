# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT WordPiece tokenizer (lower case)."""
from __future__ import annotations

import unicodedata

NEVER_SPLIT = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")


def whitespace_tokenize(text: str) -> list:
    text = text.strip()
    return text.split() if text else []


class BertTokenizer:
    def __init__(self, vocab: list, max_input_chars_per_word: int = 100):
        self.vocab = {}
        for i, token in enumerate(vocab):
            self.vocab[token] = i
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> list:
        return [piece for token in self._basic_tokenize(text) for piece in self._wordpiece(token)]

    def convert_tokens_to_ids(self, tokens: list) -> list:
        unk = self.vocab["[UNK]"]
        return [self.vocab.get(t, unk) for t in tokens]

    # basic tokenization: clean up, lower case, strip accents, split off punctuation
    def _basic_tokenize(self, text: str) -> list:
        text = "".join(" " if _is_whitespace(c) else c for c in text
                       if not (ord(c) == 0 or ord(c) == 0xFFFD or _is_control(c)))
        text = "".join(f" {c} " if _is_chinese_char(ord(c)) else c for c in text)
        tokens = []
        for token in whitespace_tokenize(text):
            if token not in NEVER_SPLIT:
                token = "".join(c for c in unicodedata.normalize("NFD", token.lower())
                                if unicodedata.category(c) != "Mn")
            tokens.extend(_split_on_punc(token))
        return whitespace_tokenize(" ".join(tokens))

    # greedy longest-match-first WordPiece
    def _wordpiece(self, token: str) -> list:
        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return ["[UNK]"]
        pieces, start = [], 0
        while start < len(chars):
            end, piece = len(chars), None
            while start < end:
                sub = "".join(chars[start:end])
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    piece = sub
                    break
                end -= 1
            if piece is None:
                return ["[UNK]"]
            pieces.append(piece)
            start = end
        return pieces


def _split_on_punc(text: str) -> list:
    if text in NEVER_SPLIT:
        return [text]
    output, start_new_word = [], True
    for c in text:
        if _is_punctuation(c):
            output.append([c])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(c)
    return ["".join(x) for x in output]


def _is_chinese_char(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _is_whitespace(c: str) -> bool:
    return c in " \t\n\r" or unicodedata.category(c) == "Zs"


def _is_control(c: str) -> bool:
    return c not in "\t\n\r" and unicodedata.category(c).startswith("C")


def _is_punctuation(c: str) -> bool:
    cp = ord(c)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(c).startswith("P")
