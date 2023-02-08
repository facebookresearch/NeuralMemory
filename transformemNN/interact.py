import torch
import torch.nn.functional as F
from droidlet.memory.filters_conversions import new_filters_to_sqly
import sys
from prettytable import PrettyTable
import ast
from transformers import GPT2Tokenizer
from collections import OrderedDict
import math

TIMESTEP = 0


def build_2clause_q():
    claus1 = input("Clause 1: ")
    claus2 = input("Clause 2: ")
    conjunction = input("Conjunction: ")

    reqs = [claus1, claus2]

    clauses = []
    for req in reqs:
        if "name" in req:
            clauses.append({"pred_text": "has_name", "obj_text": req})
        else:
            clauses.append({"pred_text": "has_tag", "obj_text": req})
    w = clauses.pop()
    where = {}
    while clauses:
        # if torch.rand(1).item() < not_prob:
        #     w = {"NOT": [w]}
        if conjunction.upper() == "AND":
            w = {"AND": [w, clauses.pop()]}
        else:
            w = {"OR": [w, clauses.pop()]}

    query = {"output": "MEMORY", "where_clause": w, "memory_type": "ReferenceObject"}
    return query


def parse_lf(input_dict, cur_str):
    for k, v in input_dict.items():
        for clause in v:
            if "not" in clause:
                sys.stdout.write("not ")
                parse_lf(clause, cur_str)
            else:
                print(
                    clause["input_left"]["value_extractor"]["attribute"],
                    clause["comparison_type"],
                    clause["input_right"]["value_extractor"],
                )


def build_spatial_query():
    attribute = input("attribute: ")
    comparison_type = input("comparison_type (LESS_THAN or GREATER_THAN): ")
    value = input("value: ")
    query_text = (
        "{'output': 'MEMORY', 'where_clause': {'AND': [{'input_left': {'value_extractor': {'attribute': '"
        + attribute
        + "'}}, 'comparison_type': '"
        + comparison_type
        + "', 'input_right': {'value_extractor': "
        + str(value)
        + "}}]}, 'memory_type': 'ReferenceObject'}"
    )
    return query_text


def format_query(query_tokens, tokenizer):
    query_text = tokenizer.decode(query_tokens)
    query_text = [x for x in query_text if "[CLS]" not in x]
    query_text = [x for x in query_text if "[SEP]" not in x]
    query_text = [x for x in query_text if "[PAD]" not in x]
    query_text = " ".join(query_text)
    query_text = query_text.replace(" _ ", "_")
    query_text = query_text.replace(" ( ", " (")
    query_text = query_text.replace(" ) ", ") ")
    query_text = query_text.replace(" ##", "")
    query_text = query_text.replace("' ", "'").replace(" '", "'")
    query_text = ast.literal_eval(query_text)
    where_clause = query_text["where_clause"]

    print("QUERY:")

    print(where_clause)
    # parse_lf(where_clause,'')

    return where_clause


def format_refobjs(word_tokens, tokenizer):
    token_list = []
    for tok in word_tokens:
        tok = tokenizer.convert_ids_to_tokens(tok.item())
        if tok not in ["!", "none"]:
            tok = tok.replace(" ", "")
            token_list.append(tok)

    raw_words = " ".join(token_list)
    raw_words = raw_words.replace("inst _ se g", "inst_seg")
    raw_words = raw_words.replace("inst se g", "inst_seg")
    raw_words = raw_words.replace("!", " ")
    raw_words = raw_words.replace("          ", "")
    return raw_words


def format_triples(word_tokens, tokenizer):
    raw_words = " ".join(tokenizer.convert_ids_to_tokens(word_tokens))
    raw_words = raw_words.replace(" [PAD]", "").replace("[PAD]", "")
    raw_words = raw_words.replace(" _ ", "_")
    raw_words = raw_words.replace("has_tag_", "has_tag ")
    raw_words = raw_words.replace(" none", "").replace("none ", "")
    raw_words = raw_words.replace(" triple", "").replace("triple ", "")

    raw_words = raw_words.replace("!", " ").replace("None", "")
    raw_words = raw_words.replace("has_name", "").replace("has_tag", "")
    raw_words = raw_words.replace("tri ple", "").replace("mob", "")
    raw_words = raw_words.replace(" ", "")

    return raw_words


def format_tensor(tensor_in):
    tensor_out = ["%.2f" % elem for elem in tensor_in]
    return tensor_out


def format_location(tensor_in, SL):
    # COORD_SHIFT = -SL//2
    # tensor_in[tensor_in!=0] = (tensor_in[tensor_in!=0]  * (SL / 2)) + (SL / 2 + COORD_SHIFT)
    xyz_max = SL - 1
    xyz_min = 0
    # tensor_in[tensor_in!=0] = (((tensor_in[tensor_in!=0] + 1)/2)*(xyz_max-xyz_min)) + xyz_min
    tensor_in[tensor_in != 0] = (
        ((tensor_in[tensor_in != 0])) * (xyz_max - xyz_min)
    ) + xyz_min
    string_out = "({:.2f}, {:.2f}, {:.2f})".format(
        tensor_in[0], tensor_in[1], tensor_in[2]
    )

    return string_out


def format_pitch_yaw(pitch, yaw, norm=False):
    pitch = pitch.item()
    yaw = yaw.item()
    pitch_min, pitch_max = -math.pi, math.pi
    yaw_min, yaw_max = -math.pi, math.pi
    if norm:
        pitch = (((pitch)) * (pitch_max - pitch_min)) + pitch_min
        yaw = (((yaw)) * (yaw_max - yaw_min)) + yaw_min
    string_out = "({: .2f}, {: .2f})".format(pitch, yaw)

    return string_out


def get_text(tokens, tokenizer, nz_indices=None):
    if nz_indices is not None:
        tokens = torch.index_select(tokens.cpu(), 0, nz_indices.cpu())
    text = tokenizer.convert_ids_to_tokens(tokens)
    text = [x.replace("Ġ", "") for x in text]
    text = " ".join(text)
    return text


def print_memory(X, k, tokenizer, t_size, r_size, out_memid=None):
    memid_loss_k = None

    context_text = X["context"]["text"][k]
    context_text = tokenizer.decode(context_text)
    context_text = context_text.replace("!", "")  # pad tokens in GPTtokenizer
    print('\nText Memory: "{}"\n'.format(context_text))

    t = PrettyTable(
        ["T", "RefObj", "Location", "Pitch, Yaw", "Properties", "Label", "Pred"]
    )
    t.align = "l"
    t.align["Label"] = "c"

    for TIMESTEP in range(X["context"]["r_attn"].size(0)):
        ys_ref_obj_k = X["context"]["r_attn"][TIMESTEP, k, :, 0]
        if out_memid is not None:
            Y_ref_obj_k_sm = (ys_ref_obj_k / ys_ref_obj_k.sum()).cpu()
            ref_obj_mem_out_k = out_memid[
                k,
                ((TIMESTEP + 1) * t_size + TIMESTEP * r_size) : (
                    (TIMESTEP + 1) * t_size + TIMESTEP * r_size
                )
                + r_size,
            ]
            ref_obj_mem_out_k[ref_obj_mem_out_k == 0.0] = float("-inf")
            ref_obj_mem_out_k = torch.softmax(ref_obj_mem_out_k.cpu(), dim=-1)
            memid_loss_k = (
                torch.nan_to_num(
                    (Y_ref_obj_k_sm * torch.log(Y_ref_obj_k_sm / ref_obj_mem_out_k))
                )
                .sum()
                .item()
            )
            Y_ref_obj_k = ["%.2f" % elem for elem in Y_ref_obj_k_sm.tolist()]
            ref_obj_mem_out_k = ["%.2f" % elem for elem in ref_obj_mem_out_k.tolist()]

        triples_hash = X["context"]["t_hash"][TIMESTEP][k]
        triples_words = X["context"]["t_word"][TIMESTEP][k]
        triples_float = X["context"]["t_float"][TIMESTEP][k]

        ref_obj_hash = X["context"]["r_hash"][TIMESTEP][k]
        ref_obj_words = X["context"]["r_word"][TIMESTEP][k]
        ref_obj_float = X["context"]["r_float"][TIMESTEP][k]

        # loop through each ref_obj
        for r_idx in range(0, ref_obj_hash.size(0)):
            r_label = ys_ref_obj_k[r_idx].item()
            r_pred = None
            if out_memid is not None:
                r_pred = ref_obj_mem_out_k[r_idx]

            r_hash = ref_obj_hash[r_idx][0].item()
            r_words = ref_obj_words[r_idx]
            r_float = ref_obj_float[r_idx]

            r_words = format_refobjs(r_words, tokenizer)
            r_float_xyx = r_float[0:3]
            r_float_xyx = format_location(
                r_float_xyx, 15
            )  # currently rescaling by 15 to get back to xyz range
            pitch_and_yaw = format_pitch_yaw(r_float[3], r_float[4])

            if r_idx == 0:
                r_words = "none"

            row_list = [TIMESTEP, r_words, r_float_xyx, pitch_and_yaw]

            # get corresponding triple idxs
            triple_idxs = (triples_hash[:, 1] == r_hash).nonzero()
            triple_list = []

            for t_idx in triple_idxs:
                t_idx = t_idx.item()

                t_hash = triples_hash[t_idx]
                t_word = triples_words[t_idx]
                t_float = triples_float[t_idx]

                # if not a null ref_obj
                if t_hash.sum() > 0:
                    t_hash = format_tensor(t_hash)
                    t_float = format_tensor(t_float)
                    t_word = format_triples(t_word, tokenizer)
                    if t_word and (
                        t_word
                        in ["_in_others_inventory", "_on_ground", "_in_inventory"]
                        or t_word[0] != "_"
                    ):
                        triple_list.append(t_word)

            row_list.append(triple_list) if triple_list else row_list.append("")
            row_list.append("*") if r_label == 1 else row_list.append("")
            row_list.append(r_pred) if r_pred is not None else row_list.append("")

            # only add to print if not empty refobj
            if r_words.replace(" ", "") != "":
                t.add_row(row_list)

    print(t)

    return memid_loss_k


def get_preds_and_loss(args, Y, k, tokenizer, out_text):

    pred_text_k = None
    Y_text_k = None

    if out_text is not None:
        if args.text_kl:
            Y_new = Y[:, 1:]  # remove [START]
            Y_text_binary = torch.zeros(out_text.size(0), out_text.size(-1)).to(
                args.device
            )
            Y_text_binary.scatter_(dim=-1, index=Y_new.long(), value=1)
            Y_text_binary[:, 0] = 0  # ignore padding
            Y_text_binary_sm = Y_text_binary / Y_text_binary.sum(1).unsqueeze(1)

            out_text = out_text[:, 0]

            Y_k = Y_new[k]
            O_k = out_text[k]
            Y_text_binary_sm_k = Y_text_binary_sm[k]

            Y_k = Y_k[Y_k.nonzero()][:, 0]

            _, O_k_sorted = torch.sort(O_k, descending=True)

            Y_k_set = list(
                OrderedDict.fromkeys(Y_k.tolist())
            )  # remove duplicates for now because we only evaluate single tokens
            O_k_top = O_k_sorted[0 : len(Y_k_set)]
            Y_text_k = get_text(Y_k_set, tokenizer)
            out_max_k = get_text(O_k_top, tokenizer)
            text_loss_out = F.kl_div(O_k, Y_text_binary_sm_k, reduction="sum")

        else:
            Y_k = Y[k]
            Y_k = Y_k[1:]  # remove [START] BOS

            pred_text_k = out_text[k]
            out_max_k = out_text.max(dim=-1)[1][k].cpu()

            L, V = pred_text_k.shape
            pred_text_flat = pred_text_k.reshape(-1, V)
            Y_flat = Y_k.reshape(-1)
            Y_nz_idxs = (Y_flat != 0).nonzero()[:, 0]
            Y_flat = Y_flat[Y_nz_idxs]

            pred_text_flat = pred_text_flat[Y_nz_idxs]

            text_loss_out = F.nll_loss(pred_text_flat, Y_flat)

            first_eos_index = (out_max_k == tokenizer.eos_token_id).nonzero(
                as_tuple=True
            )[0]
            if len(first_eos_index) > 0:
                first_eos_index = first_eos_index[0].item()
                out_max_k[first_eos_index + 1 :] = 0  # zero all tokens past EOS
            Y_text_k = get_text(Y_k, tokenizer, Y_nz_idxs)
            out_max_k = get_text(out_max_k, tokenizer, Y_nz_idxs)

    if args.text_kl:
        match_len = len(list(set(Y_text_k) & set(out_max_k)))
        Y_len = len(Y_text_k)
    else:
        match_len = len(
            [i for i, j in zip(Y_text_k.split(" "), out_max_k.split(" ")) if i == j]
        )
        Y_len = len(Y_text_k.split(" "))

    return Y_text_k, out_max_k, text_loss_out, Y_len, match_len


def print_memory_and_results(
    args, batch_ind, k, X, Y, out_memid, out_text, t_size, r_size, tokenizer, print_err
):
    memid_loss_k = None
    Y_text_k, out_max_k, text_loss_out, Y_len, match_len = get_preds_and_loss(
        args, Y, k, tokenizer, out_text
    )

    query_k = X["question"]["query_text"][k]
    query_k = tokenizer.decode(query_k)
    query_k = query_k.replace("!", "")

    if (not print_err) or (print_err and (Y_text_k != out_max_k)):
        # if (not print_err) or (print_err and (set(Y_text_k) != set(out_max_k))):
        # if (not print_err) or (print_err and (memid_loss_k>0.1)):
        batch_sz = 16
        print("\n{}\n".format((batch_ind * batch_sz) + k))
        print('Query: "{}"'.format(query_k))
        memid_loss_k = print_memory(X, k, tokenizer, t_size, r_size, out_memid)
        if Y_text_k is not None:
            print("Answer: {}".format(Y_text_k))
            print("Prediction: {}".format(out_max_k))
            print("Text Loss : {:.2f}".format(text_loss_out.item()))
            print("Text Match: {}/{}".format(match_len, Y_len))
        if memid_loss_k is not None:
            print("MemId Loss: {:.2f}".format(memid_loss_k))

    return


def read_input_gpt(
    model, args, k, X, Y, out_memid, out_text, t_size, r_size, tokenizer
):
    print("\n")
    new_query = input("Enter query: ")
    while new_query != "":
        query_ids = torch.LongTensor(tokenizer.encode(new_query))
        X["question"]["query_text"][k].fill_(0)
        X["question"]["query_text"][k][0 : query_ids.size(0)] = query_ids
        with torch.no_grad():
            out_text, out_memid, attn, src_key_padding_mask = model(X)

        print_memory_and_results(
            args,
            k,
            X,
            Y,
            out_memid,
            out_text,
            t_size,
            r_size,
            tokenizer,
            args.print_err,
        )

        new_query = input("Enter query: ")
        print("\n\n")

    return


def read_input(model, k, X, Y, out_memid, out_text, t_size, r_size, tokenizer):
    print("\n")

    interact = input("Press i to interact: ")
    if interact == "i":

        # query_text = build_2clause_q()
        query_text = build_spatial_query()
        try:
            query_text = new_filters_to_sqly(query_text)
        except:
            query_text = str(query_text)
        query_tokens = tokenizer(query_text)
        query_tokens = torch.LongTensor(query_tokens)

        X["question"]["query_text"][k].fill_(0)
        X["question"]["query_text"][k][0 : query_tokens.size(0)] = query_tokens
        with torch.no_grad():
            out_text, out_memid, attn, src_key_padding_mask = model(X)

        if out_memid is not None:
            ref_obj_mem_out_k = out_memid[k, t_size:]

        print_memory_and_results(
            k, X, Y, out_memid, out_text, t_size, r_size, tokenizer
        )

        input("Press enter to continue: ")
        print("\n\n")

    return


def interact_batch(args, model, X, Y, batch_ind):
    if args.tokenizer == "bert":
        from utils.tokenizer import Tokenizer

        tokenizer = Tokenizer()
    elif args.tokenizer == "gpt":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        from utils.tokenizer import SimpleTextTokenizer

        tokenizer = SimpleTextTokenizer(tokenizer_path)

    with torch.no_grad():
        out_text, out_memid, attn, src_key_padding_mask = model(X)

    t_size = args.triples_buffer_size
    r_size = args.refobjs_buffer_size

    for k in range(X["question"]["query_text"].size(0)):
        # if real_loss - best_loss > 0.25:
        print_memory_and_results(
            args,
            batch_ind,
            k,
            X,
            Y,
            out_memid,
            out_text,
            t_size,
            r_size,
            tokenizer,
            args.print_err,
        )
        # if args.model_type == 'gpt':
        #     read_input_gpt(model, args, k, X, Y, out_memid, out_text, t_size, r_size, tokenizer)
        # else:
        #     read_input(model, k, X, Y, out_memid, out_text, t_size, r_size, tokenizer)
        interact = input("Press any key to continue: ")

    return
