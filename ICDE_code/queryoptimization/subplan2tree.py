# This file (1)define the tree structure (2)get the tree structure from the subquery(subplan) for step(state,obj)

class Tree():
    def __init__(self):
        # self.parent = None
        self.left = None
        self.index = None
        self.l_table = None
        self.l_table_id = None
        self.l_name = None
        self.l_column = []
        self.l_column_id = []
        self.l_column_embed = []
        self.l_table_embed = None

        self.right = None
        self.r_table = None
        self.r_table_id = None
        self.r_name = None
        self.r_column = []
        self.r_column_id = []
        self.r_column_embed = []
        self.r_table_embed = None

        self.num=None
        self.all_selectivity=None
        self.left_all_selectivity = None
        self.right_all_selectivity = None
        self.joined_num=0
        self.hint=None

    

    
    

