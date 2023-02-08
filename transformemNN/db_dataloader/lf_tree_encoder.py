class TreeNode:
    def __init__(self, text):
        self.text = text
        self.parent = None
        self.children = []
        self.index = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def __str__(self, prefix=None, last_child=False) -> str:
        if prefix is None:
            prefix = ""
            s = ""
        else:
            s = prefix + "|__"
            if last_child:
                prefix += "   "
            else:
                prefix += "|  "
        s += self.text
        if self.index is not None:
            s += f" [{self.index}]"
        s += "\n"
        for i, c in enumerate(self.children):
            if i < len(self.children) - 1:
                s += c.__str__(prefix, False)
            else:
                s += c.__str__(prefix, True)
        return s

    def set_index(self, ind2node):
        self.index = len(ind2node)
        ind2node.append(self)
        for c in self.children:
            c.set_index(ind2node)

    def traverse_relation(self, rel, limit=0):
        if self.visited:
            return
        if len(rel) > limit > 0:
            return
        self.visited = True
        self.relation = rel
        if self.parent is not None:
            self.parent.traverse_relation(rel + "P", limit)
        for c in self.children:
            c.traverse_relation(rel + "C", limit)


def lf2tree(lf, key=None):
    if type(lf) is list:
        if key is None:
            tree = TreeNode("[list]")
        else:
            tree = TreeNode(key)
        for e in lf:
            tree.add_child(lf2tree(e))
    elif type(lf) is dict:
        if key is None:
            tree = TreeNode("[dict]")
        else:
            tree = TreeNode(key)
        for k, v in lf.items():
            if k in ["AND", "OR", "NOT"]:
                assert len(lf) == 1
                if key is None:
                    # in this case, no need to create a node for the dict
                    return lf2tree(v, k)
            tree.add_child(lf2tree(v, k))
    else:
        # must be a leaf node
        assert type(lf) is str
        if key is None:
            tree = TreeNode(lf)
        else:
            tree = TreeNode(key)
            tree.add_child(lf2tree(lf))
    return tree


def lf_tree_build(lf):
    tree = lf2tree(lf, "root")
    ind2node = []
    tree.set_index(ind2node)
    node_rels = []
    for src in ind2node:
        for n in ind2node:
            n.visited = False
            n.relation = "[FAR]"
        src.traverse_relation("", limit=4)
        node_rels.append([n.relation for n in ind2node])
    return tree, ind2node, node_rels


def lf_tree_encode(lf, tokenizer):
    tree, ind2node, node_rels = lf_tree_build(lf)
    for r in node_rels:
        for i in range(len(r)):
            r[i] = tokenizer.encode(r[i])

    encoded = []
    for n in ind2node:
        encoded.append(tokenizer.encode(n.text))

    return {"encoded": encoded, "node_rels": node_rels}


if __name__ == "__main__":
    lf = {
        "output": "MEMORY",
        "where_clause": {
            "AND": [
                {"NOT": [{"pred_text": "has_tag", "obj_text": "prop_0"}]},
                {"pred_text": "has_tag", "obj_text": "prop_3"},
            ]
        },
        "memory_type": "ReferenceObject",
    }

    tree, ind2node, node_rels = lf_tree_build(lf)
    print(tree)
