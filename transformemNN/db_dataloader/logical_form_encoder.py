def is_number(x):
    return type(x) is float or type(x) is int

def shift_span(span_tuple, shift):
    return (span_tuple[0] + shift, span_tuple[1] + shift)

def extend_from_child(o, child):
    """
    takes the output struct at a node p in the linearization and
    extends it based on the output from the linearization of a child node c
    in particular, shifts all the spans from c by the length of output struct
    from p before extending p's output struct by c's output struct.
    """
    l = len(o["encoded"])
    for t in ["number_spans", "coordinate_spans", "text_spans"]:
        o[t].extend(shift_span(x, l) for x in child[t])
    o["encoded"].extend(child["encoded"])
    
def linearize_and_encode(lf, text_tokenizer, key_coder):
    """
    out: a dict with fields 
        "encoded", "text_spans", "pos_spans", "look_spans", "number_spans".
    out["encoded"] is a list of numbers, corresponding either to
        dictionary entries or floats
    out["text_spans"] is a list of (start, end) tuples delineating
        spans of encoded that were tokenized by the text tokenizer
        and need to be fed to the text embedder
    out["coordinate_spans"] is a list of (start, end) tuples delineating
        spans of encoded that correspond to [x, y, z] pos tuples
    out["number_spans"] is a list of (start, end) tuples delineating
        spans of encoded that correspond to lists of "generic" numbers
    """
    o = {"encoded":[],
         "text_spans": [],
         "coordinate_spans": [],
         "number_spans": []}
    if type(lf) is list:
        if any([type(v) is list or type(v) is dict for v in lf]):
            o["encoded"].append(key_coder("["))
            for v in lf:
                child = linearize_and_encode(v, text_tokenizer, key_coder)
                extend_from_child(o, child)
            o["encoded"].append(key_coder("]"))
        elif all([is_number(x) for x in lf]):
            o["encoded"] = key_coder.encode_numbers(lf)
            o["number_spans"] = [(0, len(o["encoded"]))]  
        else: #list of other/mixed stuff, let the text tokenizer sort it out
            o["encoded"] = text_tokenizer(str(lf))
            o["text_spans"] = [(0, len(o["encoded"]))]
    elif type(lf) is dict:
        o["encoded"].append(key_coder("{"))
        n = len(lf)
        for k, v in lf.items():
            i = key_coder(k)
            assert(i)
            o["encoded"].append(i)
            if key_coder.is_coordinates_key(k):
                c = key_coder.encode_numbers(v)
                o["coordinate_spans"].append((len(o["encoded"]), len(o["encoded"])+3))
                o["encoded"].extend(c)
            else:
                child = linearize_and_encode(v, text_tokenizer, key_coder)
                extend_from_child(o, child)
            if n > 1:
                o["encoded"].append(key_coder(","))
        o["encoded"].append(key_coder("}"))
    else:
        slf = str(lf)
        idx = key_coder(slf)
        if idx is None:
            o["encoded"] = text_tokenizer(str(lf))
            o["text_spans"] = [(0, len(o["encoded"]))]
        else:
            o["encoded"] = [idx]
    return o

def delinearize_and_decode(o, text_tokenizer, key_coder):
    # for now doesn't decode the text spans
    decoded = ["NOT_DECODED"]*len(o["encoded"])
    for t in ["number_spans", "coordinate_spans", "text_spans"]:
        for s in o[t]:
            decoded[s[0]:s[1]] = o["encoded"][s[0]:s[1]]
    for i in range(len(o["encoded"])):
        if decoded[i] == "NOT_DECODED":
            k = key_coder.decode(o["encoded"][i])
            if k not in ["{", "}", ",", "[", "]"]:
                k = '"' + k + '"' 
                k = k + ":"
            decoded[i] = k

    return decoded
        


            
if __name__ == "__main__":
    from transformemNN.utils.tokenizer import LFKeyCoder
    pos = [2,3,4]
    ctype = "GREATER_THAN"
    distance_lf = {
        "relative_direction": "AWAY",
        "source": {"reference_object": {"special_reference": {"coordinates_span": pos}}}
    }
    clause = {
        "input_left" : {"value_extractor" : {"attribute": distance_lf}},
        "comparison_type" : ctype,
        "input_right" : {"value_extractor" : 5}
    }

    w = {"AND":[clause]}
    f = {"output": "MEMORY", "where_clause": w, "memory_type": "ReferenceObject"}
    C = LFKeyCoder()
    
    def dummy_text_coder(s):
        return [10]*len(s.split())
        
    e = linearize_and_encode(f, dummy_text_coder, C)

