import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from interact import interact_batch
from utils.logger import write_granular_losses
from transformers import GPT2Tokenizer

PADDING_INDEX = 0
MAX_DECODE_LEN = 15

# FIXME move to self.tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def update_granular_losses(
    args,
    conj_type_log,
    X,
    pred_text,
    Y_text,
    Y_text_seq,
    clause_type_log,
    err_indices=None,
):
    if clause_type_log:
        queries = tokenizer.batch_decode(X["question"]["query_text"])
        and_queries = [idx for idx, s in enumerate(queries) if " and " in s]
        or_queries = [idx for idx, s in enumerate(queries) if " or " in s]
        none_queries = [
            idx
            for idx, s in enumerate(queries)
            if (not " or " in s and not " and " in s)
        ]
        if args.text_kl:
            granular_loss = F.kl_div(pred_text, Y_text, reduction="none").sum(1)
        else:
            B, L, V = pred_text.shape
            pred_text_flat = pred_text.reshape(-1, V)
            Y_flat = Y_text_seq.reshape(-1)
            Y_mask = Y_text_seq.ne(0)

            text_loss_out = F.nll_loss(pred_text_flat, Y_flat, reduction="none")
            text_loss_out = text_loss_out.view(B, L)
            text_loss_out = text_loss_out * Y_mask
            granular_loss = text_loss_out.sum(-1)

        and_kl = (
            torch.index_select(
                granular_loss, 0, torch.tensor(and_queries, device=args.device)
            )
            .sum()
            .item()
        )
        or_kl = (
            torch.index_select(
                granular_loss, 0, torch.tensor(or_queries, device=args.device)
            )
            .sum()
            .item()
        )
        none_kl = (
            torch.index_select(
                granular_loss, 0, torch.tensor(none_queries, device=args.device)
            )
            .sum()
            .item()
        )

        conj_type_log["AND"]["loss"] += and_kl
        conj_type_log["OR"]["loss"] += or_kl
        conj_type_log["NONE"]["loss"] += none_kl

        conj_type_log["AND"]["samples"] += len(and_queries)
        conj_type_log["OR"]["samples"] += len(or_queries)
        conj_type_log["NONE"]["samples"] += len(none_queries)

        for i in range(Y_text_seq.size(0)):
            if X["question"]["clause_types"] is not None:
                clause_types = X["question"]["clause_types"][i]
                for clause_type in clause_types:
                    clause_type_log["loss"][clause_type] = (
                        clause_type_log["loss"].get(clause_type, 0)
                        + granular_loss[i].item()
                    )
                    clause_type_log["count"][clause_type] = (
                        clause_type_log["count"].get(clause_type, 0) + 1
                    )
                    if err_indices is not None:
                        clause_type_log["error"][clause_type] = (
                            clause_type_log["error"].get(clause_type, 0)
                            + err_indices[i].item()
                        )
            true_tokens = Y_text_seq[i][Y_text_seq[i].nonzero()][:, 0]
            N = len(true_tokens)
            _, pred_indices_ranked = torch.sort(pred_text[i], descending=True)
            top_n_preds = pred_indices_ranked[0:N]

            if args.text_kl:
                true_tokens_set = set(true_tokens.tolist())
                if args.text_kl:
                    top_n_preds_set = set(top_n_preds.tolist())
                    if true_tokens_set == top_n_preds_set:
                        pass

            B, D = pred_text.size(0), pred_text.size(-1)  # batch and dictionary size
            Y_text_binary = torch.zeros(B, D).to(args.device)
            Y_text_binary.scatter_(dim=-1, index=Y_text_seq.long(), value=1)
            Y_text_binary[:, 0] = 0  # ignore padding


def compute_count_loss(
    args,
    X,
    pred_memid_in,
    pred_text_in,
    attns,
    src_pad_mask,
    clause_type_log,
    granular_losses=False,
):
    memid_loss_full = None
    memid_loss_ref_obj = None
    memid_loss_KL = None
    text_loss_out = None
    err_indices = None
    err = None

    t_attn = X["context"]["t_attn"]
    r_attn = X["context"]["r_attn"]
    Y_memid_in = torch.cat([t_attn, r_attn], 2)
    Y_text_in = X["answer"]

    ## MemID Supervision ##
    if pred_memid_in is not None:
        bsz = pred_memid_in.size(0)

        t_size = args.triples_buffer_size
        r_size = args.refobjs_buffer_size

        pred_memid = pred_memid_in

        Y_memid = Y_memid_in.permute(1, 0, 2, 3).reshape(bsz, -1)  # B x T_b

        s = Y_memid.sum(1)
        ys = Y_memid / s.unsqueeze(1)

        y_refobj = torch.cat(
            (
                Y_memid[:, t_size : t_size + r_size],
                Y_memid[:, ((t_size * 2) + r_size) :],
            ),
            1,
        )
        pred_memid_refobj = torch.cat(
            (
                pred_memid[:, t_size : t_size + r_size],
                pred_memid[:, (t_size * 2 + r_size) :],
            ),
            1,
        )

        src_pad_mask = src_pad_mask[:, : pred_memid.size(1)]  # remove query mask
        src_pad_mask_refobj = torch.cat(
            (
                src_pad_mask[:, t_size : t_size + r_size],
                src_pad_mask[:, (t_size * 2 + r_size) :],
            ),
            1,
        )

        ys_ref_obj = y_refobj / y_refobj.sum(1).unsqueeze(1)

        # ignore pad in the softmax
        pred_memid = pred_memid.masked_fill(src_pad_mask == 0, float("-inf"))
        pred_memid_refobj = pred_memid_refobj.masked_fill(
            src_pad_mask_refobj == 0, float("-inf")
        )

        o = F.log_softmax(pred_memid, dim=-1)
        o_ref_obj = F.log_softmax(pred_memid_refobj, dim=-1)
        o_ref_obj_sm = F.softmax(pred_memid_refobj, dim=-1)

        memid_loss_full = -(o[src_pad_mask > 0] * ys[src_pad_mask > 0]).sum() / bsz
        memid_loss_ref_obj = (
            -(
                o_ref_obj[src_pad_mask_refobj > 0] * ys_ref_obj[src_pad_mask_refobj > 0]
            ).sum()
            / bsz
        )
        memid_loss_KL = (
            torch.nan_to_num(
                (
                    ys_ref_obj[src_pad_mask_refobj > 0]
                    * torch.log(
                        ys_ref_obj[src_pad_mask_refobj > 0]
                        / o_ref_obj_sm[src_pad_mask_refobj > 0]
                    )
                )
            ).sum()
            / bsz
        )

    ## Attention Supervision ##
    total_attn_loss = None
    if args.attn_supervision:
        attn_mat = torch.stack(attns).mean(0)
        for idx, attn_mat in enumerate(attns):  # layers
            if idx > 0:  # don't use first layer
                # attn_mat =  BK x L x L
                attn_mat = attn_mat.view(
                    Y_memid.size(0), -1, attn_mat.size(-2), attn_mat.size(-1)
                )
                attn_mat_head = attn_mat.mean(1)  # mean across heads
                # for head_idx in range(attn_mat.size(1)):
                # attn_mat_head = attn_mat[:,head_idx,:,:]
                Y_memid_attn = torch.zeros(
                    attn_mat_head.shape, device=attn_mat_head.device
                )
                Y_memid_attn[:, :, 0 : Y_memid.size(1)] = Y_memid.unsqueeze(1)
                Y_memid_attn[:, :, Y_memid.size(1) :] = 1
                Y_memid_attn_norm = Y_memid_attn / Y_memid_attn.sum(2).unsqueeze(-1)

                if args.attn_supervision_type == "ce":
                    attn_loss = -(
                        torch.log(attn_mat_head) * Y_memid_attn_norm
                    ).sum() / Y_memid.size(0)
                elif args.attn_supervision_type == "kl":
                    gtz_mask = Y_memid_attn_norm > 0
                    attn_loss = (
                        Y_memid_attn_norm[gtz_mask]
                        * torch.log(
                            Y_memid_attn_norm[gtz_mask] / attn_mat_head[gtz_mask]
                        )
                    ).sum() / Y_memid.size(0)
                else:
                    attn_loss = (
                        attn_mat_head * (1 - Y_memid_attn)
                    ).sum() / Y_memid.size(0)
                if total_attn_loss is None:
                    total_attn_loss = attn_loss
                else:
                    total_attn_loss += attn_loss

    if args.memid_loss:
        if args.triple_supervision:
            mem_id_loss_out = memid_loss_full
        elif args.attn_supervision:
            bsz_times_heads = attns[0].size(0)
            layers = len(attns)
            mem_id_loss_out = memid_loss_ref_obj + (
                args.attn_supervision_weight * total_attn_loss
            )
        else:
            mem_id_loss_out = memid_loss_ref_obj

    ## Text Supervision ##
    if args.text_loss:
        # pred_text = pred_text_in[:,:-1,:] # remove [STOP] (since we don't have it in Y)
        pred_text = pred_text_in
        Y_text_seq = Y_text_in[:, 1:]  # remove [START] (B x S x D)

        if args.text_kl:
            pred_text = pred_text[
                :, 0, :
            ]  # only take first output token (there should only be 1 when using this loss)
            B, V = pred_text.shape

            # convert ground truth tokens to a binary sparse vector
            Y_text_binary = torch.zeros(B, V).to(args.device)
            Y_text_binary.scatter_(dim=-1, index=Y_text_seq.long(), value=1)
            Y_text_binary[:, 0] = 0  # ignore padding
            Y_text_binary_sm = Y_text_binary / Y_text_binary.sum(1).unsqueeze(
                1
            )  # softmax(Y)

            text_loss_out = F.kl_div(pred_text, Y_text_binary_sm, reduction="batchmean")

            update_granular_losses(
                args,
                granular_losses,
                X,
                pred_text,
                Y_text_binary_sm,
                Y_text_seq,
                clause_type_log,
            )

        else:
            pred_text = pred_text[:, :-1]  # take out the last prediction (from EOS)
            B, L, V = pred_text.shape

            pred_text_flat = pred_text.reshape(-1, V)
            Y_flat = Y_text_seq.reshape(-1)
            Y_mask = Y_text_seq.ne(0)

            # compute nll loss
            text_loss_out = F.nll_loss(
                pred_text_flat, Y_flat, reduction="none", ignore_index=0
            )
            text_loss_out = text_loss_out.reshape(B, L)
            text_loss_out = text_loss_out.sum(
                -1
            ).mean()  # only average over the batch dim

            # compute error
            _, pred = pred_text.max(dim=-1)
            err_indices = Y_text_seq.ne(pred)
            err_indices = err_indices * Y_mask
            err_indices = err_indices.any(
                1, keepdim=True
            ).float()  # if even 1 token is wrong, count it as error
            err = err_indices.mean()

            update_granular_losses(
                args,
                granular_losses,
                X,
                pred_text,
                Y_flat,
                Y_text_seq,
                clause_type_log,
                err_indices,
            )

    if args.memid_loss and args.text_loss:
        # both memid and text loss
        loss_out = args.memid_coeff * mem_id_loss_out + text_loss_out
    elif args.memid_loss:
        # only memid loss
        loss_out = mem_id_loss_out
    else:
        # only text loss
        loss_out = text_loss_out

    return (
        loss_out,
        err,
        err_indices,
        text_loss_out,
        memid_loss_full,
        memid_loss_ref_obj,
        memid_loss_KL,
    )


def analyze_error_distribution(q_types, err_indices, stat):
    for i in range(err_indices.size(0)):
        q_type = q_types[i]
        tot_k = f"{q_type}_count"
        err_k = f"{q_type}_err"
        err_v = stat.get(err_k, 0)
        for j in range(err_indices[i].size(0)):
            if err_indices[i][j] == 1:
                err_v += 1
        tot_v = stat.get(tot_k, 0) + 1
        stat[tot_k] = tot_v
        stat[err_k] = err_v


# separating batch training reduces memory usage (removes overlap?)
def train_batch(
    args,
    model,
    optimizer,
    scheduler,
    X,
    q_types,
    stat,
    stat_memid,
    clause_type_log,
    conj_type_log,
    batch_ind,
    test_only=False,
):

    pred_text, pred_memid, attns, src_pad_mask = model(X)

    (
        loss,
        err,
        err_indices,
        text_loss,
        memid_loss,
        memid_loss_ref_obj,
        memid_loss_KL,
    ) = compute_count_loss(
        args,
        X,
        pred_memid,
        pred_text,
        attns,
        src_pad_mask,
        clause_type_log,
        conj_type_log,
    )

    stat["loss"] = stat.get("loss", 0) + loss.item()

    if err is not None:
        stat["err"] = stat.get("err", 0) + err.item()

    if err_indices is not None:
        analyze_error_distribution(q_types, err_indices, stat)

    if text_loss is not None:
        stat["text_loss"] = stat.get("text_loss", 0) + text_loss.item()

    if memid_loss is not None:
        if memid_loss is not None:
            stat_memid["loss"] = stat_memid.get("loss", 0) + memid_loss.item()
        if memid_loss_ref_obj is not None:
            stat_memid["ref_obj_loss"] = (
                stat_memid.get("ref_obj_loss", 0) + memid_loss_ref_obj.item()
            )
        if memid_loss_KL is not None:
            stat_memid["ref_obj_KL"] = (
                stat_memid.get("ref_obj_KL", 0) + memid_loss_KL.item()
            )

    if not test_only:
        if args.fp16:
            from apex import amp

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (batch_ind + 1) % args.accumulation_steps == 0:
            loss = loss / args.accumulation_steps
            if args.grad_clip > 0:
                # global clip
                if args.fp16:
                    nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.grad_clip
                    )
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                if (
                    args.cosine_decay or args.exp_decay
                ) and args.lr_warmup > scheduler.last_epoch:
                    # do warm-up manually
                    for pg in optimizer.param_groups:
                        pg["lr"] = args.lr * scheduler.last_epoch / args.lr_warmup


def to_device(sample, device):
    if sample is None or type(sample) is str:
        return sample
    elif type(sample) is int:
        return sample
    elif type(sample) is list or type(sample) is tuple:
        return [to_device(e, device) for e in sample]
    elif type(sample) is dict:
        return {k: to_device(v, device) for (k, v) in sample.items()}
    else:
        return sample.to(device)


def train(args, model, optimizer, scheduler, dataloader, epoch=0, test_only=False):
    stat = dict()
    stat_memid = dict()

    conj_type_log = None
    clause_type_log = None
    if args.full_test:
        clause_type_log = {"loss": {}, "error": {}, "count": {}}
        conj_type_log = {
            "AND": {"loss": 0, "samples": 0},
            "OR": {"loss": 0, "samples": 0},
            "NONE": {"loss": 0, "samples": 0},
        }

    if test_only:
        model.eval()
        desc = "testing"
    else:
        model.train()
        optimizer.zero_grad()
        desc = "training"

    nbatches = 0
    if args.disable_tqdm:
        loader = dataloader
    else:
        loader = tqdm(dataloader, mininterval=0.5, desc=desc, leave=False, ncols=50)
    for batch_ind, samples in enumerate(loader):
        nbatches += 1
        q_types = samples["q_type"]

        samples = to_device(samples, args.device)

        X = samples
        Y = samples["answer"]

        if args.interact:
            interact_batch(args, model, X, Y, batch_ind)
        else:
            train_batch(
                args,
                model,
                optimizer,
                scheduler,
                X,
                q_types,
                stat,
                stat_memid,
                clause_type_log,
                conj_type_log,
                batch_ind,
                test_only,
            )

    if args.full_test:
        write_granular_losses(conj_type_log, clause_type_log, X)

    for stat_dict in [stat, stat_memid]:
        for k, v in stat_dict.items():
            if (
                k == "err"
                or k == "loss"
                or k == "ref_obj_loss"
                or k == "ref_obj_KL"
                or k == "text_loss"
            ):
                stat_dict[k] = v / nbatches

    return stat, stat_memid
