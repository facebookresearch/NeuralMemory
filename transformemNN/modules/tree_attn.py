import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-attention layer.
    """

    def __init__(self, head_dim, dropout, nheads):
        super().__init__()
        self.head_dim = head_dim
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)

    def compute_context_rel(self, query, attn, attn_mask=None):
        rels, emb = self.context_rel 
        # rels: B x L x L binary mask indicating triplet-refobj relations.
        # emb: the embedding vector to use. H x 1
        # duplicate for heads
        rels = rels.repeat_interleave(self.nheads, dim=0)
        B, L, _ = rels.size()


        # exclude question tokens. Assume context comes first!
        query_c = query[:, :L]  # B x L x H

        # compute the attention value (before softmax) for each token
        attn_c = torch.matmul(query_c, emb)  # B x L x 1
        attn_c = attn_c.expand(-1, -1, L)  # B x L x L

        if attn_mask is not None:
            attn_c += attn_mask

        attn_c = attn_c * rels

        # add it to attentions
        attn[:, :L, :L] += attn_c

        return attn


    def forward(self, query, key, value, question_rel, key_padding_mask=None):
        # query = B x M x H
        # key, value = B x L x H
        # question_rel : B x Lq x Lq x H
        # key_padding_mask : B x L

        # compute attention from context
        attn = torch.matmul(query, key.transpose(-1, -2))  # B x M x L

        if question_rel is not None:
            Lq = question_rel.size(1)
            # assume questions tokens come last
            query_q = query[:, -Lq:]  # B x Lq x H
            query_q = query_q.unsqueeze(-1)  # B x Lq x H x 1
            attn_q_rel = torch.matmul(question_rel, query_q)  # B x Lq x Lq x 1
            attn_q_rel = attn_q_rel.squeeze(-1)  # B x Lq x Lq
            assert query.size(1) == key.size(1), "assumed no static memory"
            attn[:, -Lq:, -Lq:] += attn_q_rel
        
        if hasattr(self, "context_rel"):
            attn = self.compute_context_rel(query, attn)

        attn = attn / math.sqrt(self.head_dim)  # B x M X L

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1)  # B x 1 x L
            attn += (1 - key_padding_mask).log()

        attn = F.softmax(attn, dim=-1)
        attn_drop = self.dropout(attn)  # B x M X L

        out = torch.matmul(attn_drop, value)  # B x M x H
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_sz, nheads, dropout=0):
        super().__init__()
        self.nheads = nheads
        self.hid_sz = hid_sz
        self.head_dim = hid_sz // nheads
        self.attn = SelfAttention(self.head_dim, dropout, nheads)
        self.proj_query = nn.Linear(hid_sz, self.head_dim * nheads, bias=False)
        self.proj_out = nn.Linear(self.head_dim * nheads, hid_sz, bias=False)
        self.proj_val = nn.Linear(hid_sz, self.head_dim * nheads, bias=False)
        self.proj_key = nn.Linear(hid_sz, self.head_dim * nheads, bias=False)
        nn.init.xavier_uniform_(self.proj_query.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.xavier_uniform_(self.proj_val.weight)
        nn.init.xavier_uniform_(self.proj_key.weight)

    def head_reshape(self, x):
        # x : L x B x KD
        K = self.nheads
        D = self.head_dim
        sz = x.size()
        x = x.view(sz[0], sz[1] * K, D)  # L x BK x D
        x = x.transpose(0, 1).contiguous()  # BK x L x D
        return x

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        assert attn_mask is None, "not supported"
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        L = query.size(0)
        B = query.size(1)


        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        if key_padding_mask is not None:
            # duplicate for heads
            key_padding_mask = key_padding_mask.repeat_interleave(self.nheads, dim=0)

        question_rel = None
        if hasattr(self, "question_rel_emb"):
            question_rel = self.question_rel_emb
            Lq = question_rel.size(0)
            question_rel = self.proj_key(question_rel)  # Lq x Lq x B x KD
            question_rel = question_rel.view(Lq, Lq, B * self.nheads, self.head_dim)  # Lq x Lq x BK x D
            question_rel = question_rel.permute(2, 0, 1, 3)  # BK x Lq x Lq x D

        out, attn = self.attn(query, key, value, question_rel, key_padding_mask=key_padding_mask)  # BK x L x D
        out = out.transpose(0, 1).contiguous()  # L x BK x D
        out = out.view(L, B, self.hid_sz)  # L x B x KD
        out = self.proj_out(out)
        return out, attn
