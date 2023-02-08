import os
import pickle
import torch
import torch.utils.data as tds
import re
import random


class MemoryDataset(tds.Dataset):
    def __init__(self, flist, tdict):
        self.flist = flist
        self.tdict = tdict

    def __getitem__(self, index):
        fname = self.flist[index]
        (
            context_encoding,
            question_encoding,
            answer_encoding,
            xyz_encoding,
            mems_encoding,
        ) = torch.load(fname)
        answer_encoding = torch.LongTensor(
            [self.tdict["w2i"].get(answer_encoding, self.tdict["w2i"]["unk"])]
        )
        q_type = fname[fname.rfind("/") + 1 : fname.find("__")]
        return (
            context_encoding,
            question_encoding,
            answer_encoding,
            xyz_encoding,
            mems_encoding,
            q_type,
        )

    def __len__(self):
        return len(self.flist)


class SimpleMemoryDataset(tds.Dataset):
    def __init__(self, args, data):
        self.data = data
        self.input_type = args.input_type

    def __getitem__(self, index):
        encoded_sample = self.data[index]
        encoded_sample["q_type"] = "memory"

        if self.input_type == "random":
            random.seed(index)
            sample_input_type = random.choice(["text", "lf", "both"])
        else:
            sample_input_type = self.input_type

        if sample_input_type == "text":
            encoded_sample["query_type_mask"] = [1, 0]
        elif sample_input_type == "lf":
            encoded_sample["query_type_mask"] = [0, 1]
        elif sample_input_type == "both":
            encoded_sample["query_type_mask"] = [1, 1]
        else:
            raise ValueError("Incorrect input type given")

        return encoded_sample

    def __len__(self):
        return len(self.data)


class Collater:
    def __init__(self, opts, train=False):
        # the sizes of the buffers for the featurizer; the collater
        # uses this to make arranging the examples into a flat buffer efficient
        self.opts = opts
        self.train_split = train

        self.types = {
            "t_hash": torch.int64,
            "t_word": torch.int64,
            "t_float": torch.float32,
            "r_hash": torch.int64,
            "r_word": torch.int64,
            "r_float": torch.float32,
            "t_mask": torch.int8,
            "r_mask": torch.int8,
            "t_attn": torch.long,
            "r_attn": torch.long,
            "text": torch.long,
            "text_mask": torch.long,
        }

        self.buffer_sizes = {
            "t_hash": opts.triples_buffer_size,
            "t_word": opts.triples_buffer_size,
            "t_float": opts.triples_buffer_size,
            "t_attn": opts.triples_buffer_size,
            "t_mask": opts.triples_buffer_size,
            "r_hash": opts.refobjs_buffer_size,
            "r_word": opts.refobjs_buffer_size,
            "r_float": opts.refobjs_buffer_size,
            "r_mask": opts.refobjs_buffer_size,
            "r_attn": opts.refobjs_buffer_size,
            "text": opts.text_buffer_size,
            "text_mask": opts.text_buffer_size,
        }

    def pad_answers(self, answers, padding_value=0.0):
        max_size = len(answers)
        max_len = max([len(q) for q in answers])
        out_tensor = answers[0].new_full((max_size, max_len), padding_value)
        pad_mask = answers[0].new_full((max_size, max_len), 0)

        for i in range(len(answers)):
            length = answers[i].size(0)
            out_tensor[i, :length] = answers[i]

        return out_tensor, pad_mask

    def pad_mems(self, memories, padding_value=0.0):
        """
        this function stacks a list of memories and dict_masks
        along a new dimension, and pads them to equal length.
        the dict_masks will be padded to 0.0
        Returns:
        out_encode: Tensor of size N x L
        encode_pad_mask: Tensor of size N x L, 1 means not padded, 0 means padded.
        out_dict_mask: Tensor of size N x L x 3 or None if no dict_masks input

        (L is the length of the longest question)
        """
        max_size = len(memories)
        max_len = max([len(q) for q in memories])
        out_tensor = memories[0].new_full((max_size, max_len), padding_value)
        pad_mask = memories[0].new_full((max_size, max_len), 0)

        for i in range(len(memories)):
            length = memories[i].size(0)
            out_tensor[i, :length] = memories[i]
            pad_mask[i, :length] = torch.ones(length)

        return out_tensor, pad_mask

    def pad(self, questions, padding_value=0.0, dict_masks=None, tree_rels=None):
        """
        this function stacks a list of questions and dict_masks
        along a new dimension, and pads them to equal length.
        the dict_masks will be padded to 0.0
        Returns:
        out_encode: Tensor of size N x L
        encode_pad_mask: Tensor of size N x L, 1 means not padded, 0 means padded.
        out_dict_mask: Tensor of size N x L x 3 or None if no dict_masks input

        (L is the length of the longest question)
        """
        max_size = len(questions)
        max_len = max([len(q) for q in questions])
        out_tensor = questions[0].new_full((max_size, max_len), padding_value)
        pad_mask = questions[0].new_full((max_size, max_len), 0)
        if dict_masks is None or dict_masks[0] is None:
            out_dict_mask = None
        else:
            out_dict_mask = dict_masks[0].new_full((max_size, max_len, 3), 0)
        if tree_rels is None or tree_rels[0] is None:
            out_tree_rels = None
        else:
            out_tree_rels = tree_rels[0].new_full((max_size, max_len, max_len), 0)
        for i in range(len(questions)):
            length = questions[i].size(0)
            out_tensor[i, :length] = questions[i]
            pad_mask[i, :length] = torch.ones(length)
            if out_dict_mask is not None:
                out_dict_mask[i, :length, :] = dict_masks[i]
            if out_tree_rels is not None:
                out_tree_rels[i, :length, :length] = tree_rels[i]
        return out_tensor, pad_mask, out_dict_mask, out_tree_rels

    def __call__(self, examples):
        """
        examples is a list, each element in the list has a
        context_encoding
        question_encoding
        answer_encoding
        memid_encoding
        question_types

        the context encoding is further broken down into
        triples_hash (LongTensor)
        triples_words (LongTensor)
        triples_float (FloatTensor)
        reference_objects_hash (LongTensor)
        reference_objects_words (LongTensor)
        reference_objects_float (FloatTensor)
        t_mask (IntTensor)
        reference_object_mask (IntTensor)
        triple_attn (FloatTensor)
        reference_object_attn (FloatTensor)

        collater groups these so that the batches have the same list/dict structure,
        but the Tensors are batched.

        Output of collated_contexts is a list containing 8 db inputs:
            and 2 attention weights for a total of 10 elements

        [
            collated_triples_hash_tensor, size: (T, B, F_t, D_0)
            collated_triples_words_tensor, size: (T, B, F_t, D_1)
            ...
            collated_reference_object_mask_tensor, size: (T, B, F_m, D_7)
            triple_attn, size: (B, F_t)
            reference_object_attn, size: (B, F_r)
        ]

        Here:

        T: number of timesteps
        B: batch size
        F_s: buffer size: s in [triples, reference_objects, memories]
        D_i: size of field i

        """
        N_example = len(examples)
        # T is the number of timesteps, assuming T is the same for all examples for now
        T = examples[0]["context"]["t_hash"].size(0)

        collated_contexts = {}
        for k in examples[0]["context"].keys():
            if len(examples[0]["context"][k].size()) > 1:
                context_dim = examples[0]["context"][k].size(2)
                c = torch.zeros(
                    (T, N_example, self.buffer_sizes[k], context_dim),
                    dtype=self.types[k],
                )
                for j in range(N_example):
                    l = examples[j]["context"][k].shape[1]
                    c[:, j, :l, :] = examples[j]["context"][k]
                collated_contexts[k] = c
            else:  # text and text mask
                a = torch.zeros(N_example, self.buffer_sizes[k])
                for j in range(N_example):
                    max_len = (
                        min(len(examples[j]["context"][k]), self.buffer_sizes[k]) - 1
                    )
                    a[j, :max_len] = examples[j]["context"][k][:max_len]
                collated_contexts[k] = a

        question_encodings_text = [e["question"]["query_text"] for e in examples]
        query_text_mask = [e["question"]["query_text_mask"] for e in examples]
        question_encodings_tlf = [e["question"]["query_tlf"] for e in examples]
        dictionary_masks = [e["question"]["masks"] for e in examples]
        tree_rels = [e["question"]["tree_rels"] for e in examples]
        clause_types = None
        if "clause_types" in examples[0]["question"]:
            clause_types = [e["question"]["clause_types"] for e in examples]

        (query_text, _, _, _) = self.pad(question_encodings_text)
        (query_text_mask, _, _, _) = self.pad(query_text_mask)
        (query_tlf, q_masks, q_dict_masks, q_tree_rels) = self.pad(
            question_encodings_tlf, dict_masks=dictionary_masks, tree_rels=tree_rels
        )

        question_dict = {
            "query_text": query_text,
            "query_text_mask": query_text_mask,
            "query_tlf": query_tlf,
            "query_slf": None,
            "masks": q_masks,
            "dict_masks": q_dict_masks,
            "tree_rels": q_tree_rels,
            "clause_types": clause_types,
        }

        if self.train_split is True:
            # shuffle ordering of tokens (for selecting one)
            answers = []
            for e in examples:
                start_char = e["answer"][0:1]
                answer = e["answer"][1:]
                answers.append(torch.cat((start_char, answer), 0))
        else:
            answers = []
            for e in examples:
                start_char = e["answer"][0:1]
                answer = e["answer"][1:]
                answers.append(torch.cat((start_char, answer), 0))

        answers, answer_mask = self.pad_answers(answers)

        mems, mem_masks, _, _ = self.pad(
            [e["mems"][0] for e in examples],
            padding_value=self.opts.uuid_hash_vocab_size,
        )

        q_types = [e["q_type"] for e in examples]
        query_type_mask = torch.Tensor([e["query_type_mask"] for e in examples])

        collated_sample = {
            "context": collated_contexts,
            "question": question_dict,
            "answer": answers,
            "mem": [mems, mem_masks, torch.stack([e["mems"][1] for e in examples])],
            "q_type": q_types,
            "query_type_mask": query_type_mask,
        }

        return collated_sample


def maybe_add_word_to_dict(d, word):
    if d["w2i"].get(word) is None:
        d["w2i"][word] = len(d["i2w"])
        d["i2w"].append(word)


def build_dict(flist):
    tdict = {"w2i": {}, "i2w": []}
    maybe_add_word_to_dict(tdict, "unk")
    for f in flist:
        e = torch.load(f)
        ans_encoding = e[2]
        maybe_add_word_to_dict(tdict, ans_encoding)
    return tdict


def get_flist(data_path):
    """
    It assumes the data_path has the following directory structure

    data_path
    |
    |-- 0
    |-- 1
    |-- ...
    |-- flist
       |
       | -- 0.pkl
       | -- 1.pkl

    where subdir data_path/0, data_path/1, ... contains data
    and data_path/flist/0.pkl, data_path/flist/1.pkl, etc. contains list of
    full file names within each corresponded dir.

    """
    flists = []
    flist_dir = f"{data_path}/flist/"
    for root, dirs, files in os.walk(flist_dir):
        for fname in files:
            flist_name = os.path.join(root, fname)
            with open(flist_name, "rb") as f:
                flist = pickle.load(f)
            flists.extend(flist)
    # if there is no |-- flist subdir, try to just read the files by hand
    if not flists:
        print("WARNING, no flist dir walking subdirectories")
        for root, dirs, files in os.walk(data_path):
            for fname in files:
                f = os.path.join(root, fname)
                flists.append(f)
    return flists


def get_data(args, data_dir):
    if args.simple_data_path:
        data = torch.load(args.simple_data_path)
        ntrain = int(len(data) * args.split_ratio)
        train_dataset = SimpleMemoryDataset(args, data[:ntrain])
        val_dataset = SimpleMemoryDataset(args, data[ntrain:])
        return train_dataset, val_dataset

    # put file names into list for lazy loading
    flist = get_flist(args.episode_dir)
    print(f"Total data num: {len(flist)}")
    print("building dictionary")
    tdict = build_dict(
        flist[:100000]
    )  # only take subset of flist to build dict for speed

    nlabels = len(tdict["i2w"])
    print(f"nlables: {nlabels}")

    print(f"i2w:")
    print(tdict["i2w"])

    train_flist = flist[: int(len(flist) * args.split_ratio)]
    val_flist = flist[int(len(flist) * args.split_ratio) :]

    train_dataset = MemoryDataset(train_flist, tdict)
    val_dataset = MemoryDataset(val_flist, tdict)

    return train_dataset, val_dataset, nlabels, tdict


def get_presplit_data(args, data_path):
    data = torch.load(data_path)
    dataset = SimpleMemoryDataset(args, data)
    return dataset


def get_training_chunk(data_dir):
    # retrieves random training chunk file
    chunk_ids = []
    patt = re.compile(r"train_\d+_\d+\.pth")
    for f in os.listdir(data_dir):
        if patt.match(f):
            f = f.replace(".pth", "")
            chunk_id = f.split("train_")[1]
            chunk_ids.append(chunk_id)

    assert len(chunk_ids) > 0

    random_id = random.choice(chunk_ids)
    train_file = os.path.join(data_dir, "train_{}.pth".format(random_id))

    return train_file


if __name__ == "__main__":
    get_data()
