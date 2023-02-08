import os
import sys
import torch
import math

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)

data_gen_path = this_path + "../../data"
sys.path.append(data_gen_path)

transformemnn_path = this_path.strip("db_dataloader")
utils_path = transformemnn_path + "/utils/"
sys.path.append(utils_path)

from tokenizer import Tokenizer, LFKeyCoder, LFSimpleTokenizer, SimpleTextTokenizer
from transformers import GPT2Tokenizer
from logical_form_encoder import linearize_and_encode
from lf_tree_encoder import lf_tree_encode
from utils.nsm_utils import UUID_HASH_VOCAB_SIZE
from constants import (
    TOKENS_PER_CELL,
    T_HASH,
    T_FLOAT,
    T_WORD,
    R_HASH,
    R_WORD,
    REFOBJ_FLOAT,
)


def hash2n(uuid, hash_vocab_size):
    # WARNING! no randomization here, assuming the uuid.hex in the agent will randomize
    # todo special number for self memid (00000000000000)
    return hash(uuid) % hash_vocab_size


def encode_text(tokenizer, text):
    text = "None" if not text else str(text)
    ids = tokenizer(text.lower(), add_special_tokens=False)
    if type(ids) is not str and type(ids) is not list:
        ids = ids["input_ids"]

    return torch.LongTensor(ids)


def encode_text_prealloc(tokenizer, tensor, text, idx):
    ids = encode_text(tokenizer, text)

    tensor[idx : idx + len(ids)] = torch.LongTensor(ids)
    return idx + len(ids)


def encode_triple(
    triple_dict, tokenizer, text_tokenizer_type, hash_vocab_size, none_time=-1
):
    S = max(list(triple_dict.keys())) + 1
    t_hash = torch.zeros((S, T_HASH), dtype=torch.long)
    t_word = torch.zeros((S, T_WORD), dtype=torch.long)
    t_float = torch.zeros((S, T_FLOAT), dtype=torch.float)
    t_mask = torch.zeros((S, 1), dtype=torch.int8)
    for s, triple_step in triple_dict.items():
        t_mask[s, 0] = 1

        t_hash[s, 0] = hash2n(triple_step["uuid"], hash_vocab_size)
        t_hash[s, 1] = hash2n(triple_step["subj"], hash_vocab_size)
        t_hash[s, 2] = hash2n(triple_step["pred"], hash_vocab_size)
        t_hash[s, 3] = hash2n(triple_step["obj"], hash_vocab_size)

        # from the mainmem table:
        t_float[s, 0] = triple_step.get("create_time") or none_time
        t_float[s, 1] = triple_step.get("updated_time") or none_time
        t_float[s, 2] = triple_step.get("attended_time") or none_time

        encode_text_prealloc(tokenizer, t_word[s], triple_step["subj_text"], 0)
        encode_text_prealloc(
            tokenizer, t_word[s], triple_step["pred_text"], TOKENS_PER_CELL
        )
        encode_text_prealloc(
            tokenizer, t_word[s], triple_step["obj_text"], 2 * TOKENS_PER_CELL
        )
        encode_text_prealloc(
            tokenizer, t_word[s], triple_step["node_type"], 3 * TOKENS_PER_CELL
        )

    return t_hash, t_word, t_float, t_mask


def encode_refobj(
    ref_obj_dict,
    tokenizer,
    text_tokenizer_type,
    hash_vocab_size,
    none_look=(0, 0),
    none_loc=(0, 0, 0),
    none_time=-1,
    SL=60,
    COORD_SHIFT=(-30, 54, -30),
):
    S = max(list(ref_obj_dict.keys())) + 1
    #    S = len(ref_obj_dict)
    r_mask = torch.zeros((S, 1), dtype=torch.int8)
    r_hash = torch.zeros((S, R_HASH), dtype=torch.long)
    r_word = torch.zeros(
        (S, R_WORD), dtype=torch.long
    )  # .fill_(220) # fill with space token from GPT2tokenizer
    r_float = torch.zeros((S, REFOBJ_FLOAT), dtype=torch.float)
    for s, ref_obj_step in ref_obj_dict.items():
        r_mask[s, 0] = 1

        # 2 hash:
        r_hash[s, 0] = hash2n(ref_obj_step["uuid"], hash_vocab_size)
        r_hash[s, 1] = hash2n(ref_obj_step["eid"], hash_vocab_size)

        # 5 words:
        encode_text_prealloc(tokenizer, r_word[s], ref_obj_step["name"], 0)
        encode_text_prealloc(
            tokenizer, r_word[s], ref_obj_step["type_name"], TOKENS_PER_CELL
        )
        encode_text_prealloc(
            tokenizer, r_word[s], ref_obj_step["ref_type"], 2 * TOKENS_PER_CELL
        )
        encode_text_prealloc(
            tokenizer, r_word[s], ref_obj_step["node_type"], 3 * TOKENS_PER_CELL
        )
        player_placed = "player_placed" if ref_obj_step["player_placed"] else "none"
        encode_text_prealloc(tokenizer, r_word[s], player_placed, 4 * TOKENS_PER_CELL)

        # Normalize xyz to [0, 1]
        x = ref_obj_step["x"] or none_loc[0]
        y = ref_obj_step["y"] or none_loc[1]
        z = ref_obj_step["z"] or none_loc[2]
        xyz_min, xyz_max = 0, SL - 1
        r_float[s, 0] = round(x)
        r_float[s, 1] = round(y)
        r_float[s, 2] = round(z)

        # normalize pitch/yaw to [0, 1]
        pitch = ref_obj_step.get("pitch") or none_look[0]
        yaw = ref_obj_step.get("yaw") or none_look[1]
        pitch_min, pitch_max = -math.pi, math.pi
        yaw_min, yaw_max = -math.pi, math.pi
        r_float[s, 3] = pitch
        r_float[s, 4] = yaw

        # from the mainmem table:
        r_float[s, 5] = ref_obj_step.get("create_time") or none_time
        r_float[s, 6] = ref_obj_step.get("updated_time") or none_time
        r_float[s, 7] = ref_obj_step.get("attended_time") or none_time

        r_float[s, 8] = ref_obj_step["voxel_count"] or 0.0

        r_float[s, 9] = ref_obj_step["bounding_box"][0] - ref_obj_step["x"]
        r_float[s, 10] = ref_obj_step["bounding_box"][1] - ref_obj_step["x"]
        r_float[s, 11] = ref_obj_step["bounding_box"][2] - ref_obj_step["y"]
        r_float[s, 12] = ref_obj_step["bounding_box"][3] - ref_obj_step["y"]
        r_float[s, 13] = ref_obj_step["bounding_box"][4] - ref_obj_step["z"]
        r_float[s, 14] = ref_obj_step["bounding_box"][5] - ref_obj_step["z"]

    return r_hash, r_word, r_float, r_mask


def remove_memory_entities(db_dump, removable_names):
    ## Remove RefObjs that are in removeable_names and their corresponding triples
    removeable_memids = []
    for memid in list(db_dump["context"]["reference_objects"].keys()):
        for T in db_dump["context"]["reference_objects"][memid].keys():
            ref_type = db_dump["context"]["reference_objects"][memid][T]["ref_type"]
            if ref_type in removable_names:
                db_dump["context"]["reference_objects"].pop(memid, None)
                removeable_memids.append(memid)
                break
    for key in list(db_dump["context"]["triples"].keys()):
        for T in db_dump["context"]["triples"][key].keys():
            if db_dump["context"]["triples"][key][T]["subj"] in removeable_memids:
                db_dump["context"]["triples"].pop(key, None)
                break

    return db_dump


class DBEncoder:
    def __init__(
        self,
        none_time=-1,
        none_look=(0, 0),
        none_loc=(0, 0, 0),
        query_encode="text",
        SL=60,
        COORD_SHIFT=(-30, 54, -30),
        tokenizer="bert",
        tokenizer_path=None,
        remove_instseg_locations=False,
    ):
        if tokenizer == "bert":
            self.text_tokenizer = Tokenizer()
        elif tokenizer == "gpt":
            self.text_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            self.text_tokenizer = SimpleTextTokenizer(tokenizer_path)

        self.tokenizer_type = tokenizer
        self.query_encode = query_encode
        if self.query_encode == "seq_lf":
            self.lf_tokenizer = LFKeyCoder()
        elif self.query_encode == "tree_lf":
            self.lf_tokenizer = LFSimpleTokenizer()
        self.uuid_hash_vocab_size = UUID_HASH_VOCAB_SIZE
        self.none_time = none_time
        self.none_look = none_look
        self.none_loc = none_loc
        self.tokenizer = tokenizer
        self.SL = SL
        self.COORD_SHIFT = COORD_SHIFT
        self.remove_instseg_locations = remove_instseg_locations

    def hash2n(self, uuid):
        # WARNING! no randomization here, assuming the uuid.hex in the agent will randomize
        # todo special number for self memid (00000000000000)
        return hash(uuid) % self.uuid_hash_vocab_size

    def encode_context(self, context_d, memids, triple_memids):
        """
        outputs:
        t_hash: num_triples x num_time_steps x (4 + 1).  "4" corresponds to
                uuids of self, subj, pred, obj (hashed); +1 is for "is this memory attended" (binary)
        t_word: num_triples x num_time_steps x T_WORD
        t_float: num_triples x num_time_steps x T_FLOAT
        r_hash: num_refobj x num_time_steps x (2 + 1).  "2" corresponds to uuid and eid of self,
                + 1 is for "is this memory attended" (binary)
        r_word: num_refobj x num_time_steps x R_WORD.  the tokens are
                name, type_name, ref_type, node_type (from basemem table) player_placed.  player_placed is encoding of
                player name or "none"
        r_float: num_refobj x num_time_steps x REFOBJ_FLOAT float.  the float fields are
                 x,y,z,yaw, pitch, create_time, update time, attended_time, voxel count, and 6 floats for bbox

        t_mask: num_triples x num_time_steps x 1. binary: does the triple exist at that time step?
        r_mask: num_ref_obj x num_time_steps x 1. binary: does the ref_obj exist at that time step?

        triple_attn: num_triples. binary: did agent attend to this triple?
        ref_obj_attn: num_ref_obj. binary: did agent attend to this ref_obj?
        """
        S = context_d["num_steps"]

        #################
        # Triples
        triples = context_d["triples"]
        num_triples = len(triples)
        t_mask = torch.ones((S, num_triples, 1), dtype=torch.int8)

        t_attn = torch.zeros((S, num_triples, 1), dtype=torch.long)
        # eventually do confidence too?
        t_hash = torch.zeros((S, num_triples, T_HASH), dtype=torch.long)
        t_word = torch.zeros((S, num_triples, T_WORD), dtype=torch.long)
        t_float = torch.zeros((S, num_triples, T_FLOAT), dtype=torch.float)

        i = 0
        for uuid, d in triples.items():
            if uuid in triple_memids:
                t_attn[:, i, :] = 1

            h, w, f, m = encode_triple(
                d,
                self.text_tokenizer,
                self.tokenizer_type,
                self.uuid_hash_vocab_size + 1,
            )
            t_hash[:, i, :] = h
            t_word[:, i, :] = w
            t_float[:, i, :] = f
            t_mask[:, i, :] = m
            i += 1

        #################
        # reference objects
        ref_objs = context_d["reference_objects"]

        num_ref_obj = len(ref_objs)
        r_mask = torch.zeros((S, num_ref_obj, 1), dtype=torch.int8)

        r_attn = torch.zeros((S, num_ref_obj, 1), dtype=torch.long)
        r_hash = torch.zeros((S, num_ref_obj, R_HASH), dtype=torch.long)
        r_word = torch.zeros((S, num_ref_obj, R_WORD), dtype=torch.long)
        r_float = torch.zeros((S, num_ref_obj, REFOBJ_FLOAT), dtype=torch.float)

        i = 0
        for uuid, d in ref_objs.items():
            if uuid in memids:
                r_attn[:, i, :] = 1
            h, w, f, m = encode_refobj(
                d,
                self.text_tokenizer,
                self.tokenizer_type,
                self.uuid_hash_vocab_size + 1,
                SL=self.SL,
                COORD_SHIFT=self.COORD_SHIFT,
            )
            r_hash[:, i, :] = h
            r_word[:, i, :] = w
            r_float[:, i, :] = f
            r_mask[:, i, :] = m
            i += 1

        #################
        # null attn flag (if no relevant mems exist)
        # there are other ways to do this, but for now I"ll append it
        # to the start of the r_attn vec
        null_attn = torch.zeros((S, 1, 1), dtype=torch.long)
        if t_attn.sum().item() == 0 and r_attn.sum().item() == 0:
            null_attn[:, 0] = 1
        r_attn = torch.cat((null_attn, r_attn), 1)

        null_hash = torch.zeros(r_hash.size(0), 1, r_hash.size(2))
        null_word = torch.zeros(r_word.size(0), 1, r_word.size(2))
        null_float = torch.zeros(r_float.size(0), 1, r_float.size(2))
        null_mask = torch.ones(r_float.size(0), 1, r_mask.size(2))
        null_hash[:, :, -1].fill_(
            UUID_HASH_VOCAB_SIZE
        )  # Need this to differentiate from other tokens

        null_float[:, :, 3] = 0  # pitch
        null_float[:, :, 4] = 0  # yaw

        r_hash = torch.cat((null_hash, r_hash), 1)
        r_word = torch.cat((null_word, r_word), 1)
        r_float = torch.cat((null_float, r_float), 1)
        r_mask = torch.cat((null_mask, r_mask), 1)

        # Get the "text" version of the context. This is a text form that can be read by a language model
        text = context_d["text"]
        if self.tokenizer == "gpt":
            text_encoding = encode_text(self.text_tokenizer, text)
        else:
            text_encoding = torch.zeros(len(text.split(" "))).long()
            encode_text_prealloc(self.text_tokenizer, text_encoding, text, 0)

        text_mask = torch.ones(text_encoding.shape)

        context_dict = {
            "t_hash": t_hash,
            "t_word": t_word,
            "t_float": t_float,
            "r_hash": r_hash,
            "r_word": r_word,
            "r_float": r_float,
            "t_mask": t_mask,
            "r_mask": r_mask,
            "t_attn": t_attn,
            "r_attn": r_attn,
            "text": text_encoding,
            "text_mask": text_mask,
        }
        return context_dict

    def encode_question(self, db_dump):
        """
        returns a tensor x of length l with the encoded query,
        and a mask m of size l x 3.
        m[i,0] is 1 if the i"th entry of x uses the number dict
        m[i,1] is 1 if the i"th entry of x uses the text dict
        m[i,2] is 1 if the i"th entry of x uses the logical form dict
        """

        query_text, query_text_mask, x_tlf, x_slf, masks, tree_rels = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if self.query_encode == "seq_lf":
            qlf = db_dump["question_logical_form"]
            q = linearize_and_encode(qlf, self.text_tokenizer, self.lf_tokenizer)
            # WARNING: this is clipping float numbers. should allow float encodings.
            x_slf = torch.LongTensor(q["encoded"])
            masks = torch.zeros(len(x_slf), 3)
            for s in q["coordinate_spans"]:
                # SL + 1 because we assume any number is bigger than -SL, and
                # 0 is the pad idx.
                x_slf[s[0] : s[1]] = x_slf[s[0] : s[1]] + self.SL + 1
                masks[s[0] : s[1], 0] = 1
            for s in q["number_spans"]:
                x_slf[s[0] : s[1]] = x_slf[s[0] : s[1]] + self.SL + 1
                masks[s[0] : s[1], 0] = 1
            for s in q["text_spans"]:
                masks[s[0] : s[1], 1] = 1
            # the plain lf mask...
            masks[:, 2] = 1 - torch.max(masks[:, 1], masks[:, 0])
        elif self.query_encode == "tree_lf":
            qlf = db_dump["question_logical_form"]
            q = lf_tree_encode(qlf, self.lf_tokenizer)
            x_tlf = torch.LongTensor(q["encoded"])
            tree_rels = torch.LongTensor(q["node_rels"])

        qtext = db_dump["question_text"]
        if self.tokenizer == "gpt":
            query_text = encode_text(self.text_tokenizer, qtext)
        else:
            query_text = torch.LongTensor(self.text_tokenizer(qtext.lower()))

        query_text_mask = torch.ones(query_text.shape)

        question = {}
        question["query_text"] = query_text if query_text is not None else None
        question["query_text_mask"] = (
            query_text_mask if query_text_mask is not None else None
        )
        question["query_tlf"] = x_tlf if x_tlf is not None else None
        question["query_slf"] = x_slf if x_slf is not None else None
        question["masks"] = masks if masks is not None else None
        question["tree_rels"] = tree_rels if tree_rels is not None else None
        question["clause_types"] = db_dump["sample_clause_types"]

        return question

    def encode_answer(self, answer_text):
        if self.tokenizer == "gpt":
            ans = encode_text(self.text_tokenizer, answer_text)
        else:
            ans = torch.LongTensor(self.text_tokenizer(answer_text.lower()))

        ans_mask = torch.ones(ans.shape)

        return ans, ans_mask

    def encode_attended_mems(self, memids_and_vals):
        # for now ignore vals...
        hashed_mems = torch.LongTensor([self.hash2n(x) for x in memids_and_vals[0]])
        attention_map = torch.zeros(self.uuid_hash_vocab_size + 1, dtype=torch.int64)
        attention_map[hashed_mems] = 1
        return hashed_mems, attention_map

    def encode_all(self, db_dump, memids=True):
        if memids:
            memids = db_dump["memids_and_vals"][0]
            triple_memids = db_dump["triple_memids_and_vals"][0]
        else:
            memids, triple_memids = [], []

        if self.remove_instseg_locations:
            # removeable_entities = ['inst_seg','location','attention','block','11']
            removeable_entities = ["location", "attention"]
            db_dump = remove_memory_entities(db_dump, removeable_entities)

        context_encoding = self.encode_context(
            db_dump["context"], memids, triple_memids
        )
        question = self.encode_question(db_dump)
        answer_encoding, answer_mask = self.encode_answer(db_dump["answer_text"])
        mems_encoding = self.encode_attended_mems(db_dump["memids_and_vals"])

        encoded_sample = {
            "context": context_encoding,
            "question": question,
            "answer": answer_encoding,
            "answer_mask": answer_mask,
            "mems": mems_encoding,
        }

        return encoded_sample
