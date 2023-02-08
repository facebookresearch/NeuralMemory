from __future__ import print_function
from utils.tokenizer import Tokenizer, SimpleTextTokenizer


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
