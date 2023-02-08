import torch
import torch.nn as nn
from random import randrange
from utils.constants import RXYZ, RPY

# 4 hash, 4 words, 1 float
# TRIPLES_COLUMNS = 4 + 28 + 1
TRIPLES_COLUMNS = 4 + 28

# 2 hash, 5 words, 5 float (1 xyz (from 3), 1 pitch/yaw (from 2), 1 time (from 3), 1 voxel_count (from 1) 1 bbox (from 6))
# REF_OBJ_COLUMNS = 2 + 35 + 5
REF_OBJ_COLUMNS = 2 + 35 + 2


def build_fc(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ALl of these can be independent, note inputs are not scaled the same for each:


def build_pitchyaw_featurizer(args, dims):
    szs = [dims] + [args.embedding_dim] * args.num_fc_layers
    return build_fc(szs)


def build_xyz_featurizer(args):
    szs = [3] + [args.embedding_dim] * args.num_fc_layers
    return build_fc(szs)


def build_voxelcount_featurizer(args):
    szs = [1] + [args.embedding_dim] * args.num_fc_layers
    return build_fc(szs)


def build_bbox_featurizer(args):
    szs = [6] + [args.embedding_dim] * args.num_fc_layers
    return build_fc(szs)


def build_basemem_float_featurizer(args):
    szs = [3] + [args.embedding_dim] * args.num_fc_layers
    return build_fc(szs)


def convert_to_sinecos(pitchyaw, device):
    """
    Assumes input is in range [0, 2pi]

    pitchyaw[:,0] (Bx1) pitch
    pitchyaw[:,1] (Bx1) yaw
    """
    pitchyaw = pitchyaw.repeat(1, 2)[:, torch.LongTensor([0, 2, 1, 3])]

    pitchyaw[:, 0] = torch.sin(pitchyaw[:, 0])  # sin(pitch)
    pitchyaw[:, 1] = torch.cos(pitchyaw[:, 0])  # cos(pitch)

    pitchyaw[:, 2] = torch.sin(pitchyaw[:, 1])  # sin(yaw)
    pitchyaw[:, 3] = torch.cos(pitchyaw[:, 1])  # cos(yaw)

    return pitchyaw


class Featurizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding_dim = args.embedding_dim
        self.buffer_sizes = [
            args.triples_buffer_size,
            args.refobjs_buffer_size,
            args.basems_buffer_size,
        ]
        # NOTE I'm changing the lf and number vocab size because the entire query
        # gets run through the nn.embedding regardless of whether its a number or not
        # we can fix that in the forward pass by zeroing out the non-numbers in the number_emb for example
        self.vocab_size = args.vocab_size
        self.lf_vocab_size = args.vocab_size
        self.number_vocab_size = args.vocab_size
        self.uuid_hash_vocab_size = args.uuid_hash_vocab_size
        self.normalize_floats = args.normalize_floats
        self.q_lf_emb = nn.Embedding(
            self.lf_vocab_size, args.embedding_dim, padding_idx=0
        )
        self.q_number_emb = nn.Embedding(
            self.number_vocab_size, args.embedding_dim, padding_idx=0
        )

        self.hash_embedding = nn.Embedding(
            self.uuid_hash_vocab_size + 1, args.embedding_dim, padding_idx=0
        )

        if args.normalize_floats:
            pitchyaw_dim = 4
        else:
            pitchyaw_dim = 2
        self.pitchyaw = build_pitchyaw_featurizer(args, pitchyaw_dim)
        self.voxelcount = build_voxelcount_featurizer(args)
        self.bbox = build_bbox_featurizer(args)
        self.xyz = build_xyz_featurizer(args)
        self.basemem_float = build_basemem_float_featurizer(args)

        self.device = args.device
        if args.memory_unit == "cell":
            raise NotImplementedError
        else:
            # stride is the same as number of columns, so converts each block into a single embedding
            self.triple_row_combiner = torch.nn.Conv1d(
                args.embedding_dim,
                args.embedding_dim,
                TRIPLES_COLUMNS,
                TRIPLES_COLUMNS,
                bias=False,
            )
            self.refobj_row_combiner = torch.nn.Conv1d(
                args.embedding_dim, args.embedding_dim, REF_OBJ_COLUMNS, REF_OBJ_COLUMNS
            )
            # the conv acts as a blockwise linear, so column embeddings probably unnecessary.
            # no "table embeddings" either, the tables are featurized differently enough from eachother it shouldn't be problem

        self.word_emb = nn.Embedding(self.vocab_size, args.embedding_dim, padding_idx=0)
        self.hash_emb = nn.Embedding(self.vocab_size, args.embedding_dim, padding_idx=0)

        self.context_type = args.context_type
        self.model_type = args.model_type
        self.max_seq_len = args.max_seq_len

    def embed_context_text(self, text):
        text_emb = self.word_emb(text.long())
        return text_emb

    def embed_answers(self, answers):
        out = []
        for i in range(answers.size(1)):
            out.append(self.word_emb(answers[:, i].view((-1, 1))))
        return torch.cat(out, 1)

    def embed_questions(self, questions, q_type_mask):
        q_tree_rels = None
        qw = None
        qlf = None

        if questions["query_slf"]:
            # seq_lf query
            # don't use "query_text" if using seq_lf

            qn = self.q_number_emb(questions["query_slf"])
            d = qn.shape[-1]
            qn = qn * questions["masks"][:, :, 0].unsqueeze(2).expand(-1, -1, d)

            qlf = self.q_lf_emb(questions["query_slf"])
            qlf = qlf * questions["masks"][:, :, 2].unsqueeze(2).expand(-1, -1, d)

            qw = self.word_emb(questions["query_slf"])
            embedded_question = qw
            qn = qn * questions["masks"][:, :, 1].unsqueeze(2).expand(-1, -1, d)

            embedded_question = qw + qlf + qn

            return embedded_question, None, q_tree_rels

        else:
            if questions["query_text"] is not None:
                qw = self.word_emb(questions["query_text"])
                qw = qw * q_type_mask[:, 0].view(-1, 1, 1)

            if questions["query_tlf"] is not None:
                qlf = self.q_lf_emb(questions["query_tlf"])
                qlf = qlf * q_type_mask[:, 1].view(-1, 1, 1)

                q_tree_rels = self.q_lf_emb(questions["tree_rels"])
                q_tree_rels = q_tree_rels * q_type_mask[:, 1].view(-1, 1, 1, 1)

            return qw, qlf, q_tree_rels

    def hash2relation(self, t_hash, r_hash, t_mask, r_mask):
        """
        Given triple and ref-obj hash, find out which memory is connected to which memory.
        Limited to "subject_hash" in triple to "uuid" in ref-obj.

        t_hash = triple hash: T x B x triples_buffer x 4
        r_hash = reference object hash: T x B x refobj_buffer x 2
        """
        T, B, _, _ = t_hash.size()
        # select subject and uuid hash only
        t_hash = t_hash[:, :, :, 1]  # T x B x tb
        r_hash = r_hash[:, :, :, 0]  # T x B x rb

        # merge them into a single context
        hash = torch.cat([t_hash, r_hash], dim=-1)  # T x B x b (tb+rb)
        mask = torch.cat([t_mask, r_mask], dim=-2).squeeze(-1)  # T x B x b

        # merge time dimension to memory one so we connect memories across time
        hash = hash.permute(1, 0, 2).reshape(B, -1)  # B x T_b
        mask = mask.permute(1, 0, 2).reshape(B, -1)  # B x T_b

        # remove relations with padding tokens
        # set pad hash to -1, -2 so it doesn't match to anything
        hash1 = hash.masked_fill(mask == 0, -1)
        hash2 = hash.masked_fill(mask == 0, -2)

        hash1 = hash1.unsqueeze(-1)  # B x T_b x 1
        hash2 = hash2.unsqueeze(-2)  # B x 1 x T_b

        relations = hash1.eq(hash2)  # B x T_b x T_b
        return relations

    def forward(self, m):
        """
        inputs a dict m with the batched context memories & masks collated as follows:
        m["context"] has 10 items, first 6 items are context memories
            m["context"]["t_hash"] = triple hash: T x B x triples_buffer x 4
            m["context"]["t_word"] = triple words: T x B x triples_buffer x 4 * TOKENS_PER_CELL
            m["context"]["t_float"] = triple float: T x B x triples_buffer x 3
            m["context"]["r_hash"] = reference object hash: T x B x refobj_buffer x 2
            m["context"]["r_word"] = reference object words: T x B x refobj_buffer x 5 * TOKENS_PER_CELL
            m["context"]["r_float"] = reference object floats: T x B x refobj_buffer x 15
        next 2 items are time masks, the non-zero positions will be ignored while the zero
        positions will be unchanged:
            m["context"]["t_mask"] = triple mask: T x B x triples_buffer
            m["context"]["r_mask] = reference object mask: T x B x refobj_buffer
        e.g. If there are 10 snapshots and an reference object is added in the 5th snapshot
        the mask will be (along the time dimension):
                            r_mask
        Time dimension  1 1 1 1 0 0 0 0 0 0
        last 2 items are attention maps:
            m["context"]["t_attn"] = triple attention: B x triples_buffer
            m["context"]["r_attn"] = reference object attention:  B x refobj_buffer

        here
        T is the total number of steps (number of snapshoted timestamps) in the batch
        B is the batchsize
        that is: each of these are batched such that the first dimension is the time dimension,
        the second dimension is all the rows of the table across all the examples in the batch.
        The index identifying which row belongs to which example is in m["answer"].

        m["answer"] is the questions for each example; it is a B Tensor

        the final output of this is an T x B x total_buffer_size x args.embedding_dim Tensor, where
        total_buffer_size is args.triples_buffer_size + args.refobjs_buffer_size + args.basems_buffer_size
        note that the zero padding is not just at the "end" of the output Tensor, as each buffer has
        separate zero padding.  so for a single example, the output is like

                triples rows           ref_obj rows            basemem rows
        T0  * * * *     * 0 0     0 | * *     * 0 0 ... 0 | *     * 0 0     0
            * * * * ... * 0 0 ... 0 | * * ... * 0 0 ... 0 | * ... * 0 0 ... 0
            * * * *     * 0 0     0 | * *     * 0 0 ... 0 | *     * 0 0     0

        T1  * * 0 *     * 0 0     0 | * *     * 0 0 ... 0 | *     * 0 0     0
            * * 0 * ... * 0 0 ... 0 | * * ... * 0 0 ... 0 | * ... * 0 0 ... 0
            * * 0 *     * 0 0     0 | * *     * 0 0 ... 0 | *     * 0 0     0

        here the * columns are the embeddings of the rows from the db, zeros are padding.
        there are B of these arrays. Note that the third column in T1 is also all 0s.
        These are masked zeros instead of padding, which means this triple is deleted at T1.

        """
        context_rel = None
        triple_embs = None
        refobj_embs = None
        q_text_emb = None
        q_lf_emb = None
        q_text_rels = None
        answer_emb = None
        decoder_mask = None
        tgt_mask = None

        # Context
        t_float = m["context"]["t_float"].clone()
        t_word = m["context"]["t_word"].clone()
        t_hash = m["context"]["t_hash"].clone()  # TxBxStx4
        t_mask = m["context"]["t_mask"].clone()
        r_word = m["context"]["r_word"].clone()
        r_hash = m["context"]["r_hash"].clone()  # TxBxSrx2
        r_float = m["context"]["r_float"].clone()
        r_mask = m["context"]["r_mask"].clone()
        context_text = m["context"]["text"].clone()
        context_text_mask = m["context"]["text_mask"].clone()

        # Query
        questions = m["question"]
        q_type_mask = m["query_type_mask"]
        q_text_raw = questions["query_text"][
            :, 0 : self.max_seq_len
        ]  # clip text at max_seq_len
        q_text_mask = questions["query_text_mask"]

        # Answer
        answers = m["answer"]
        if answers is not None:
            tgt_mask = answers == 0
            tgt_mask = tgt_mask.type(torch.bool)

        # Misc
        try:
            SL = m["world_configs"]["SL"]
        except:
            SL = 15

        # we don't need to featurize the hashes etc if we're not using gpt
        if self.model_type not in ["gpt", "gpt_medium"]:
            if self.context_type == "triple_refobj_rel":
                context_rel = self.hash2relation(t_hash, r_hash, t_mask, r_mask)
                # disable hash
                t_hash.fill_(0)
                r_hash.fill_(0)
            else:
                if self.training:
                    # randomize all the hashes across samples
                    randval = randrange(self.uuid_hash_vocab_size)

                    t_hash[t_hash > 0] += randval
                    t_hash %= self.uuid_hash_vocab_size

                    r_hash[r_hash > 0] += randval
                    r_hash %= self.uuid_hash_vocab_size

            # T: time steps, B: batch size, F; buffer size, W: feature dim
            # triples:
            T, B, Ft, Wht = t_hash.size()  # Wht = num hash cells
            T_, B_, F_, Wwt = t_word.size()  # Wwt = num word cells * TOKENS_PER_CELL
            Tf_, Bf_, Ff_, Wft = t_float.size()  # Wft = float dim (3)
            assert T == T_ == Tf_ and B == B_ == Bf_ and Ft == F_ == Ff_
            # (T*B*Ft) x embedding_dim x num hash cells:
            triple_hash = (
                self.hash_embedding(t_hash.reshape(-1, Wht))
                .permute(0, 2, 1)
                .contiguous()
            )
            # (T*B*Ft) x embedding_dim x num word cells:
            triple_words = self.word_emb(t_word.reshape(-1, Wwt)).permute(0, 2, 1)
            # (T*B*Ft) x embedding_dim x 5:
            triple_float = self.basemem_float(t_float.reshape(-1, Wft)).unsqueeze(2)
            triple_stack = torch.cat(
                [
                    triple_hash,
                    triple_words,
                    # triple_float #WARNING need to add 1 more to TRIPLES_COLUMNS if adding this back
                ],
                2,
            )
            # T x B x Ft x embedding_dim:
            triple_embs = (
                self.triple_row_combiner(triple_stack).squeeze().view(T, B, Ft, -1)
            )

            # ref objs:
            T_, B_, Fr, Whr = r_hash.size()  # Whr = num hash cells
            Tw_, Bw_, Fw_, Wwr = r_word.size()  # Wwr = num word cells * TOKENS_PER_CELL
            Tf_, Bf_, Ff_, Wfr = r_float.size()  # Wfr = float dim (15)
            assert T == T_ == Tw_ == Tf_ and B == B_ == Bw_ == Bf_ and Fr == Fw_ == Ff_

            # (T*B*Fr) x embedding_dim x num hash cells:
            refobj_hash = (
                self.hash_embedding(r_hash.view(-1, Whr)).permute(0, 2, 1).contiguous()
            )

            # (T*B*Fr) x embedding_dim x num word cells:
            refobj_words = self.word_emb(r_word.view(-1, Wwr)).permute(0, 2, 1)

            r_float_2d = r_float.view(-1, Wfr)  # (T*B*Fr) x Wf
            # (T*B*Fr) x embedding_dim x 5:

            xyz = r_float_2d[:, RXYZ]
            pitchyaw = r_float_2d[:, RPY]

            if self.normalize_floats:
                # xyz = (xyz - 0)/(SL-1 - 0) # Normalize to [0,1]
                pitchyaw = convert_to_sinecos(pitchyaw, self.device)

            refobj_float = torch.stack(
                [
                    self.xyz(xyz),
                    self.pitchyaw(pitchyaw),
                    # self.basemem_float(r_float_2d[:, RBM]), #WARNING need to add 3 more to REF_OBJ_COLUMNS if adding these back
                    # self.voxelcount(r_float_2d[:, RVOL]),
                    # self.bbox(r_float_2d[:, RBBOX]),
                ],
                2,
            )
            # T X B x Fr X embedding_dim:
            refobj_embs = (
                self.refobj_row_combiner(
                    torch.cat([refobj_hash, refobj_words, refobj_float], 2)
                )
                .squeeze()
                .view(T, B, Fr, -1)
            )

            q_text_emb, q_lf_emb, q_text_rels = self.embed_questions(
                questions, q_type_mask
            )

            if answers is not None:
                answer_emb = self.embed_answers(answers)
                # don't let the text decoder attend to future tokens
                # non-zero positions are not allowed to attend. zero positions remain unchanged.
                ans_sz = answer_emb.size(1)
                decoder_mask = torch.triu(
                    torch.ones(ans_sz, ans_sz, device=self.device), diagonal=1
                )
                decoder_mask = decoder_mask.type(torch.bool)

        featurized_data = {
            "triple_embs": triple_embs,
            "refobj_embs": refobj_embs,
            "q_text_raw": q_text_raw,
            "q_text_emb": q_text_emb,
            "q_text_rels": q_text_rels,
            "q_lf_emb": q_lf_emb,
            "q_type_mask": q_type_mask,
            "q_text_mask": q_text_mask,
            "answers": answers,
            "answer_emb": answer_emb,
            "decoder_mask": decoder_mask,
            "tgt_mask": tgt_mask,
            "context_rel": context_rel,
            "context_text": context_text,
            "context_text_mask": context_text_mask,
            "t_mask": t_mask,
            "r_mask": r_mask,
        }
        return featurized_data
