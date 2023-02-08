from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modules.db_featurizer import Featurizer
from models.layers import TransformerEncoderLayer, TransformerDecoderLayer
from models.utils import get_vocab_size


# SD: source length of dynamic memory
# SS: source length of static memory
# SQ: source length of question
# S = SD + SQ
# N: batch size
# E: embed size


class TextDecoder(nn.Module):
    """
    Text prediction given context memory and query embeddings
    """
    def __init__(self, embedding_dim, nhead, dropout, activation, normalize_before, dictionary_len):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder = TransformerDecoder(num_layers=2, d_model=embedding_dim, 
                                          dim_feedforward=embedding_dim*2,
                                          nhead=nhead, dropout=dropout,
                                          activation=activation,
                                          normalize_before=normalize_before)
                                          
        self.probe = nn.Linear(embedding_dim, dictionary_len, bias=True)

    def forward(self, encoder_out, answer_embs, src_pad_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        answer_len = answer_embs.size(1)
        decoder_out = self.decoder(answer_embs.permute(1,0,2), 
                                   encoder_out.permute(1,0,2), 
                                   memory_key_padding_mask=src_pad_mask,
                                   tgt_mask=tgt_mask, 
                                   tgt_key_padding_mask=tgt_key_padding_mask
                                  )
        decoder_out = decoder_out.permute(1,0,2)
        preds = self.probe(decoder_out[:,-answer_len:,:])

        return preds


class MemoryRetriever(nn.Module):
    """
    Predicts the binary memid values
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.probe = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, src):
        B, F, d = src.size()
        vals = self.probe(src.reshape(B * F, d)).reshape(B, F)
        return vals


class Transformer(nn.Module):
    def __init__(
        self,
        args,
        d_model=19,
        nhead=5,
        vocab_size=32522,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.5,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        
        self.device = args.device
        self.text_loss = args.text_loss
        self.memid_loss = args.memid_loss
        self.featurizer = Featurizer(args)
        self.d_model = d_model
        self.context_type = args.context_type
        self.text_kl = args.text_kl

        # text encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers,
            self.d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )


        self.text_decoder = TextDecoder(self.d_model, nhead, dropout, activation, normalize_before,dictionary_len=vocab_size)

        
        self.memory_retriever = MemoryRetriever(self.d_model)
        self.nhead = nhead

        self.type_emb = nn.Embedding(5, self.d_model, padding_idx = 0)
        self.pos_encoder = nn.Embedding(4096,self.d_model)

        if self.context_type == 'triple_refobj_rel':
            head_dim = self.d_model // nhead
            self.context_rel_emb = nn.parameter.Parameter(torch.randn(head_dim, 1))

        self.time_emb = nn.Embedding(args.max_timesteps, self.d_model)
        self.use_lf = False
        

    def forward(self, src, query_embed=None, pos_embed=None):
        featurized_data = self.featurizer(src)

        triple_embs = featurized_data["triple_embs"]
        refobj_embs = featurized_data["refobj_embs"]
        context_rel = featurized_data["context_rel"]
        q_text_emb = featurized_data["q_text_emb"]
        q_lf_emb = featurized_data["q_lf_emb"]
        q_text_rels = featurized_data["q_text_rels"]
        q_type_mask = featurized_data["q_type_mask"]
        q_text_mask = featurized_data["q_text_mask"]
        answer_emb = featurized_data["answer_emb"]
        decoder_mask = featurized_data["decoder_mask"]
        tgt_mask = featurized_data["tgt_mask"]
        t_mask = featurized_data["t_mask"]
        r_mask = featurized_data["r_mask"]

        timesteps,bsz,_,emb_sz = triple_embs.shape

        # Add type embeddings to each of: ref_objs, triples, query_text, query_lf
        triple_type_emb = self.type_emb(torch.ones((triple_embs.size(1),triple_embs.size(2)),device=self.device).long()*1)
        refobj_type_emb = self.type_emb(torch.ones((refobj_embs.size(1),refobj_embs.size(2)),device=self.device).long()*2)
        query_text_type_emb = self.type_emb(torch.ones((q_text_emb.size(0),q_text_emb.size(1)),device=self.device).long()*3)
        query_lf_type_emb = self.type_emb(torch.ones((q_lf_emb.size(0),q_lf_emb.size(1)),device=self.device).long()*4)

        triple_embs += triple_type_emb
        refobj_embs += refobj_type_emb

        q_text_emb += (query_text_type_emb*q_type_mask[:,0].view(-1,1,1))
        q_lf_emb += (query_lf_type_emb*q_type_mask[:,1].view(-1,1,1))

        # Get context embedding
        context_embed_flat = torch.cat((triple_embs,refobj_embs),2)
        timesteps_range = torch.arange(timesteps,device=self.device).long()
        timesteps_emb = self.time_emb(timesteps_range).view(timesteps,1,1,emb_sz)
        context_embed_flat += timesteps_emb
        context_embed_flat = context_embed_flat.permute(1,0,2,3).reshape(bsz,-1,emb_sz)

        context_pad_mask_flat = torch.cat((t_mask,r_mask),2).squeeze(-1) # T x bsz x Tr+Ro
        context_pad_mask_flat = context_pad_mask_flat.permute(1,0,2).reshape(bsz,-1) # bsz x T*(Tr+Ro)

        N, SD, E = context_embed_flat.size()
        N, SQ, E = q_text_emb.size()
        
        ##### Text Input #####          
        position_id = torch.arange(q_text_emb.size(1), dtype=torch.long, device=q_text_emb.device).view(1,-1)
        pos_emb = self.pos_encoder(position_id).repeat(q_text_emb.size(0),1,1)
        q_text_emb = q_text_emb + pos_emb
        
        if context_rel is not None:
            for lay in self.encoder.layers:
                lay.self_attn.attn.context_rel = (context_rel, self.context_rel_emb)
        
        
        # At some point, we may want to tell the encoder where the text and LF inputs are
        # it works for now because the LF encoder uses the last N positions for the positional embeddings
        if self.use_lf:
            #### LF Input #####
            for lay in self.encoder.layers:
                lay.self_attn.q_text_rels = q_text_rels.permute(1, 2, 0, 3)
            # q_type_mask[:,0] are the text queries
            # q_type_mask[:,1] are the lf queries
            ## note, these are not mutually exclusive. One sample may have both text and lf
            src_embed = torch.cat((context_embed_flat, q_text_emb, q_lf_emb), 1) 
            q_text_pad_mask = q_type_mask[:, 0:1].repeat(1, q_text_emb.size(1))
            q_lf_pad_mask = q_type_mask[:, 1:2].repeat(1, q_lf_emb.size(1))
            src_key_padding_mask = torch.cat((context_pad_mask_flat, q_text_pad_mask, q_lf_pad_mask), 1)
        else:
            src_embed = torch.cat((context_embed_flat, q_text_emb), 1)  
            src_embed = src_embed.permute(1, 0, 2)  # SxNxE

            src_key_padding_mask = torch.cat((context_pad_mask_flat, q_text_mask), 1)
        
        src_key_padding_mask_flipped = 1-src_key_padding_mask

        static_mem = None # we can use this if we want to do some other static embedding...
        model_out, attns = self.encoder(src_embed, static_mem, src_key_padding_mask=src_key_padding_mask_flipped)
        model_out = model_out.permute(1, 0, 2)
        model_out = model_out*(src_key_padding_mask.unsqueeze(-1))

        ##### Memid predictions #####
        memory_out = model_out[:,0:SD]
        pred_memid = self.memory_retriever(memory_out)
        
        ##### Text predictions #####
        pred_text = None
        if self.text_loss:
            if self.text_kl: 
                # predict all tokens simultanously from BOS token (i.e., non-sequential)
                answer_emb = answer_emb[:,0:1,:] 
                tgt_mask = tgt_mask[:,0:1]
                decoder_mask = decoder_mask[0:1,0:1]
            pred_text = self.text_decoder(model_out,answer_emb,src_key_padding_mask_flipped,decoder_mask,tgt_mask)
            pred_text = F.log_softmax(pred_text, dim=-1)

        return pred_text, pred_memid, attns, src_key_padding_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers, d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, normalize_before
                )
            )

        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model) if normalize_before else None

    def forward(
        self,
        dynamic,
        static,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = dynamic
        attns = []
        for layer in self.layers:
            output,attn = layer(
                output, static, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
            attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        normalize_before,
        return_intermediate=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation, normalize_before
                )
            )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model) if normalize_before else None
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


def build_transformer(args,vocab_size):
    return Transformer(
        args,
        d_model=args.embedding_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        vocab_size=vocab_size,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        activation=args.activation,
    )