import torch
import torch.nn.functional as F
from torch import nn, Tensor
from modules.db_featurizer import Featurizer
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config

class Transformer(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        
        self.device = args.device
        self.text_loss = args.text_loss
        self.memid_loss = args.memid_loss
        self.featurizer = Featurizer(args)
        self.text_kl = args.text_kl
        
        if args.pretrained_gpt:
            if args.model_type == 'gpt_medium':
                print('GPT MEDIUM')
                self.gpt = GPT2LMHeadModel.from_pretrained("gpt2-medium")
            else:
                print('GPT SMALL')
                self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            configuration = GPT2Config()
            self.gpt = GPT2LMHeadModel(configuration)

        # self.gpt.resize_token_embeddings(args.vocab_size)

    def forward(self, src, query_embed=None, pos_embed=None):
        featurized_data = self.featurizer(src)
        questions = featurized_data["q_text_raw"]
        query_text_mask = featurized_data["q_text_mask"]
        answers = featurized_data["answers"]
        tgt_mask = featurized_data["tgt_mask"]
        text = featurized_data["context_text"]
        context_text_mask = featurized_data["context_text_mask"]

        if self.text_kl:
            answers = answers[:,0:1]
            tgt_mask = tgt_mask[:,0:1]

        answer_len = answers.size(1)
        src = torch.cat((text,questions,answers),1)

        tgt_mask_flipped = 1-tgt_mask.int()
        attention_mask = torch.cat((context_text_mask,query_text_mask,tgt_mask_flipped),1)

        model_out = self.gpt(src.long(),attention_mask=attention_mask)['logits']

        pred_text = model_out[:,-answer_len:,:]

        return F.log_softmax(pred_text, dim=-1), None , None, None


def build_gpt(args):
    return Transformer(
        args,
    )
