if __name__ == "__main__":
    import os
    import sys

    this_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(this_path)
    droidlet_path = this_path.strip("data") + "fairo/"
    sys.path.append(droidlet_path)
from nsm_utils import k_multinomial
from memory_utils import add_snapshot, convert_memory_simple, check_inactive
from config_args import get_opts
from queries_filters_based import FiltersQA
from queries_geometric_minimax import GeometricQA
from queries_geometric_basic import BasicGeometricQA
from queries_temporal_snapshot import TemporalSnapshotQA
from queries_agent_action import AgentActionQA
from queries_object_tracking import ObjectTrackingQA


# FIXME/TODO? allow multiple questions per agent_memory,
# and/or serialize/deserializing memory
class QAZoo:
    """
    this is a container class for QA objects.
    it implements a get_qa() method that instantiates
    an agent memory and a QA object, executes the QA object's
    .get_all() method, and then packages those results with
    the memory dump

    the output db_dump of the self.get_qa method has fields supplied by QA object:
    db_dump = {
        "question_text": QA.get_question_text(),
        "question_logical_form": QA.get_question_lf(), # default "NULL"
        "answer_text": QA.get_answer_text(),
        "memids_and_vals": QA.get_memids_and_vals(), # default ([], None)
        "triple_memids_and_vals": QA.get_triple_memids_and_vals(), # default ([NULL_MEMID], None)
        "sample_clause_types": QA.get_sample_clause_types(),
        "sample_conjunction_type": QA.get_sample_conjunction_type(),
    }
    and fields
    db_dump["context"] = snapshots or add_snapshot({}, convert_memory_simple(memory))
    db_dump["sample_query_type"] = qa_name
    supplied by this object.

    """

    def __init__(self, opts, configs):
        self.qa_objects = {
            "FiltersQA": FiltersQA,
            "TemporalSnapshotQA": TemporalSnapshotQA,
            "GeometricQA": GeometricQA,
            "BasicGeometricQA": BasicGeometricQA,
            "AgentActionQA": AgentActionQA,
            "ObjectTrackingQA": ObjectTrackingQA,
            # ...
        }
        self.configs = configs
        self.opts = opts

    def get_qa(self, memory=None, snapshots=None):
        qa_configs = self.configs.get("QA")
        assert qa_configs
        qprobs = {
            qname: qa_configs.get(qname, {}).get("prob", 0.0)
            for qname in qa_configs.keys()
        }
        qa_name = k_multinomial(qprobs)
        data = {"opts": self.opts, "config": qa_configs.get(qa_name, {})}
        qa_obj = self.qa_objects[qa_name](data, memory=memory, snapshots=snapshots)
        if not qa_obj.question_text:
            return False
        db_dump = qa_obj.get_all()

        if check_inactive(qa_name, qa_obj, db_dump):
            # one of the clauses is inactive
            return False
        snapshots = snapshots or add_snapshot({}, convert_memory_simple(memory))
        db_dump["context"] = snapshots
        db_dump["sample_query_type"] = qa_name
        return db_dump


if __name__ == "__main__":
    from memory_utils import build_memory

    opts, configs = get_opts()

    Z = QAZoo(opts, configs)
    memory = build_memory(opts)
    db_dump = Z.get_qa(memory=memory)
