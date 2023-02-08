from utils.nsm_utils import NULL_MEMID
from query_objects import QA


class AgentActionQA(QA):
    """
    QA object asks what agent did.
    warning: uses the lf/text, rather than actual agent actions
    to get answer
    """

    def __init__(self, data, agent=None, memory=None, snapshots=None):
        super().__init__(data, agent=agent, memory=memory, snapshots=snapshots)
        assert snapshots is not None
        chatstr = snapshots.get("chatstr", "")
        lf = snapshots.get("lf", None)
        self.question_text = " what did you do?"
        self.memids_and_vals = ([], None)
        self.sample_clause_types = []
        self.sample_conjunction_type = None
        self.question_logical_form = "NULL"
        self.triple_memids_and_vals = ([NULL_MEMID], None)
        if lf is None or len(lf) == 0:
            self.answer_text = "nothing"
        else:
            if "go around" in chatstr:
                self.answer_text = "go around"
            #                self.answer_text = snapshots.chatstr.replace(" the", "")
            elif "follow" in chatstr:
                self.answer_text = "follow"
            elif "move to" in chatstr:
                self.answer_text = "move to"
            #               self.answer_text = snapshots.chatstr
            elif "dig" in chatstr:
                self.answer_text = "dig"
            elif "destry" in chatstr:
                self.answer_text = "destroy"
            else:
                self.answer_text = "nothing"

        self.answer_text = "<|endoftext|>{}<|endoftext|>".format(self.answer_text)
