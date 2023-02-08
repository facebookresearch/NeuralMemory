class QA:
    """
    base class for QA objects.  on init or on get_all at least 
    one of question_lf or question_text should be returned,
    as well as at least one of answer_text or memids_and_vals
    """
    def __init__(self, data, agent=None, memory=None, snapshots=None):
        self.data = data
        self.agent = agent
        self.snapshots = snapshots
        self.memory = memory
        self.question_lf = {}
        self.question_text = ""
        self.answer_text = ""
        self.memids_and_vals = (None, None)
        
    def get_question_text(self):
        return self.question_text
    
    def get_question_lf(self):
        return self.question_lf

    def get_answer_text(self):
        return self.answer_text

    def get_memids_and_vals(self):
        return self.memids_and_vals

    def get_triple_memids_and_vals(self):
        return self.triple_memids_and_vals

    def get_sample_clause_types(self):
        return self.sample_clause_types

    def get_sample_conjunction_type(self):
        return self.sample_conjunction_type

    def get_all(self):
        db_dump = {
            "question_text": self.get_question_text(),
            "question_logical_form": self.get_question_lf(),
            "answer_text": self.get_answer_text(),
            "memids_and_vals": self.get_memids_and_vals(),
            "triple_memids_and_vals": self.get_triple_memids_and_vals(),
            "sample_clause_types": self.get_sample_clause_types(),
            "sample_conjunction_type": self.get_sample_conjunction_type(),
        }
        return db_dump



