"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from transformers import BertTokenizer
import os
import torch


class Tokenizer:
    def __init__(self, tokenizer=None):
        self.tokenizer = (
            BertTokenizer.from_pretrained("bert-base-uncased")
            if not tokenizer
            else tokenizer
        )
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cache = {}

    def __call__(self, text, add_special_tokens=True):
        if text not in self.cache.keys():
            out = self.tokenizer(text)["input_ids"]
            self.cache[text] = out
        return self.cache[text] if add_special_tokens else self.cache[text][1:-1]

    def text_to_ids(self, text):
        return self.tokenizer(text)["input_ids"]

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def batch_decode(self, sequence, skip_special_tokens=False):
        return self.tokenizer.batch_decode(
            sequence, skip_special_tokens=skip_special_tokens
        )

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


LF_KEYS = [
    "{",
    "}",
    "[",
    "]",
    ",",
    "reference_object",
    "text_span",
    "special_reference",
    "coordinates_span",
    "filters",
    "output",
    "contains_coreference",
    "memory_type",
    "selector",
    "return_quantity",
    "ordinal",
    "location",
    "same",
    "where_clause",
    "AND",
    "OR",
    "NOT",
    "reference_object",
    "linear_extent",
    "relative_direction",
    "frame",
    "fixed_value",
    "player_span",
    "has_measure",
    "source",
    "destination",
    "argval",
    "polarity",
    "quantity",
    "input_left",
    "value_extractor",
    "comparison_type",
    "input_right",
    "comparison_measure",
    "set_comparison",
    "modulus",
    "close_tolerance",
    "pred_text",
    "pred",
    "obj_text",
    "obj",
    "subj_text",
    "subj",
    "num_blocks",
    "block_filters",
    "attribute",
    "task_info",
]


class LFKeyCoder:
    def __init__(self):
        self.key_dictionary = {}
        self.key_dictionary["i2w"] = LF_KEYS
        self.key_dictionary["w2i"] = {LF_KEYS[i]: i for i in range(len(LF_KEYS))}

    def __call__(self, k):
        return self.key_dictionary["w2i"].get(k)

    def decode(self, i):
        return self.key_dictionary["i2w"][i]

    def is_coordinates_key(self, k):
        return k == "coordinates_span"

    def encode_numbers(self, c):
        if type(c) is list:
            return [float(x) for x in c]
        else:
            return [float(c)]


class LFSimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.ind2word = []

    def add_word(self, word):
        self.word2idx[word] = len(self.ind2word)
        self.ind2word.append(word)

    def encode_word(self, word):
        if word not in self.word2idx:
            self.add_word(word)
        return self.word2idx[word]

    def encode(self, words):
        if type(words) is str:
            return self.encode_word(words)
        inds = []
        for w in words:
            inds.append(self.encode_word(w))
        return inds


def exists(*files):
    return all([os.path.exists(file) for file in files])


class SimpleTextTokenizer:
    def __init__(self, load_path=None):
        if load_path and exists(
            load_path + "/word2idx.pth", load_path + "/word2idx.pth"
        ):
            print("=> loading tokenizer from {}".format(load_path))
            self.word2idx = torch.load(load_path + "/word2idx.pth")
            self.ind2word = torch.load(load_path + "/ind2word.pth")
        else:
            self.word2idx = {}
            self.ind2word = []
            self.add_word("[PAD]")
            self.add_word("[UNK]")
            self.add_word("[START]")
            self.add_word("[END]")
            # hacky, sorry:
            self.add_word("get")
            self.add_word("move")

    def __call__(self, words, add_special_tokens=False):
        if type(words) is str:
            return self.encode_string(words)
        else:
            return self.encode_list(words)

    def add_word(self, word):
        self.word2idx[word] = len(self.ind2word)
        self.ind2word.append(word)

    def encode_word(self, word):
        if word not in self.word2idx:
            self.add_word(word)
        return self.word2idx[word]

    def encode_string(self, string):
        inds = []
        for word in string.split(" "):
            inds.append(self.encode_word(word))
        return inds

    def encode_list(self, list):
        inds = []
        for word in word_list:
            inds.append(self.encode_word(word))
        return inds

    def convert_ids_to_tokens(self, ids):
        words = []
        for word_id in ids:
            words.append(self.ind2word[word_id])
        return words

    def batch_convert_ids_to_tokens(self, batch_ids):
        batch_list = []
        for ids in batch_ids:
            batch_list.append(convert_ids_to_tokens(ids))

        return batch_list

    def __len__(self):
        return len(self.word2idx)

    def save(self, save_dir):
        torch.save(self.word2idx, os.path.join(save_dir, "word2idx.pth"))
        torch.save(self.ind2word, os.path.join(save_dir, "ind2word.pth"))


def get_vocab_size(args):
    if args.tokenizer == "gpt":
        vocab_size = 50257
    else:
        if not args.tokenizer_path:
            args.tokenizer_path = "/".join(args.simple_data_path.split("/")[0:-1])
        try:
            tokenizer = SimpleTextTokenizer(args.tokenizer_path)
            vocab_size = len(tokenizer)
        except FileNotFoundError:
            print("loading BERT tokenizer")
            tokenizer = Tokenizer()
            vocab_size = tokenizer.vocab_size()

    return vocab_size
