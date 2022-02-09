import numpy as np
from numpy.core.fromnumeric import sort
from numpy.lib.shape_base import column_stack
from queryoptimization.subplan2tree import Tree
import queryoptimization.utils as utils
import torch
import ast
import numpy

join_conditions = {}
join_conditions_id = {}


class Query_Init():
    global join_conditions, join_conditions_id
    join_conditions = {}
    join_conditions_id = {}
    mask = []

    def __init__(self, sqlquery, schema, indices):
        global join_conditions, join_conditions_id

        schema_one_hot = np.zeros(sum(len(col)
                                      for _, col in schema.items()), dtype=int)
        relations = {}
        relations_id = {}
        pointer = 0
        column_id = {}
        id_idx = 0
        # print(sqlquery)

        self.sql_id = sqlquery.split('|')[0]
        self.table_num = len(sqlquery.split('|')[1].split('WHERE')[0].split('FROM')[1].split(" , "))
        sql = sqlquery.split('|')[1].split('$$')[0].strip(";")

        self.old_sql = sql.replace("IMDB", "AND") + ";"
        selectivity_value = sqlquery.split('|')[1].split('$$')[1]
        selectivity_value = ast.literal_eval(selectivity_value)
        selectivity_clause = sqlquery.split('|')[1].split('$$')[2]
        selectivity_clause = ast.literal_eval(selectivity_clause)
        # print(schema)
        for val, col in schema.items():
            relations[val] = list(schema_one_hot.copy())  # {a:[all_column_len]}
            relations_id[val] = id_idx  # {a:0
            id_idx += 1
            # relations_cum[val] = pointer
            for i in range(0, len(col)):
                column_name = val + '.' + col[i]
                column_id[column_name] = pointer  # {a.col=1,}
                relations[val][pointer] = 1
                pointer += 1
        self.actions = self.get_Actions(
            sql, relations, relations_id, schema, indices, column_id)

        join_conditions, join_conditions_id, self.link_mtx, self.sql_mask = self.get_Conditions(
            sql, selectivity_value, selectivity_clause, self.actions, relations_id, column_id)

        # self.actions, join_conditions, join_conditions_id, self.link_mtx = self.get_Actions_Conditions(sqlquery, relations, relations_id, schema, indices, column_id)

        for i in range(len(self.actions), len(schema)):
            self.actions.append(EmptyQuery(schema_one_hot))

        # sorted action space
        sorted_actions = []
        for key, value in schema.items():
            flag = True
            for action in self.actions:
                name = action.name
                if name == key:
                    sorted_actions.append(action)
                    flag = False
                    break

            if flag: sorted_actions.append(EmptyQuery(schema_one_hot))
        self.actions = sorted_actions



    def get_Actions(self, sql, masks, relations_id, schema, indices, column_dict):
        action_space = []

        try:
            relations = sql.split('FROM')[1].split(
                'WHERE')[0].replace(" ", "").split(',')
        except Exception:
            relations = sql.split('FROM')[1].replace(" ", "")
            pass

        for r in relations:
            r = r.replace("AS", " AS ")
            r_split = r.split(" AS ")
            # orig_table_name = r_split[0]
            join_table_name = r_split[-1]  # if no AS, then the final
            action_space.append(Relation(
                r, join_table_name, masks[join_table_name], schema[join_table_name], relations_id[join_table_name],
                indices, column_dict))
            # sql_mask+= np.array(masks[join_table_name])

        return action_space


    def get_Conditions(self, sql, selectivity_value, selectivity_clause, action_space, relations_id, column_id):
        num_relations = len(relations_id)
        link_mtx = torch.zeros((num_relations, num_relations))
        try:
            all_join_conditions = sql.split(' WHERE ')[1].split(' IMDB ')[1].split(' AND ')
        except Exception:
            all_join_conditions = []
            pass

        sql_mask = np.zeros((len(column_id)))
        # print(selectivity_value)
        # print(selectivity_clause)
        for select in selectivity_value:
            # print("select",select)
            selectivity = selectivity_value[select]
            clause = selectivity_clause[select]  #
            # print(clause)
            # selectivity = self.get_selectivity_from_clause(select,clause)
            table_name = select.split('.')[0]
            for action in action_space:

                if action.name == table_name:
                    action.set_clause(clause)
                    action.set_selectivity(select, selectivity)
                for k, v in action.embed.items():
                    sql_mask[k] = v[1]

        for condition in all_join_conditions:
            if "=" in condition:
                clauses = condition.split(' = ')
                element = []
                for clause in clauses:
                    element.append(clause.replace(
                        "\n", "").replace(" ", "").split("."))
                try:
                    l_table = element[0][0]
                    r_table = element[1][0]
                    l_col = element[0][0] + '.' + element[0][1]
                    r_col = element[1][0] + '.' + element[1][1]
                    # if relations_id[l_table] > relations_id[r_table]:
                    #     l_table, r_table = r_table, l_table
                    #     l_col, r_col = r_col, l_col

                    cond_tables = '&'.join(
                        sorted([l_table] + [r_table]))
                    try:
                        l_rel_id = relations_id[l_table]
                        r_rel_id = relations_id[r_table]
                        l_id = column_id[l_col]
                        r_id = column_id[r_col]
                        temp = 0
                        for action in action_space:
                            if action.id == l_rel_id:
                                action.set_join(l_id)
                                temp += 1
                            elif action.id == r_rel_id:
                                action.set_join(r_id)
                                temp += 1
                            if temp == 2:
                                continue

                    except Exception:
                        pass
                    join_conditions[cond_tables] = [l_col, r_col]
                    join_conditions_id[cond_tables] = [l_id, r_id]
                    link_mtx[l_rel_id][r_rel_id] = 1.0
                    link_mtx[r_rel_id][l_rel_id] = 1.0


                except Exception:
                    pass
        return join_conditions, join_conditions_id, link_mtx, sql_mask


class Relation(object):
    id = None
    name = ''
    sql_name = ''
    mask = []
    columns = []
    clauses = []

    def __init__(self, name, tab_name, mask, columns, table_id, indices, column_dict):
        self.name = tab_name
        self.sql_name = name
        self.mask = mask
        self.id = table_id
        self.indices = []
        self.columns = []  # abc.id...
        self.clauses = []
        self.selectivity = {}
        self.embed = {}
        self.column_dict = column_dict


        for column in columns:
            column_name = tab_name + '.' + column
            self.columns.append(column_name)
            self.embed[column_dict[column_name]] = [column_dict[column_name], 1, 0, table_id]

        for i in indices:
            if " AS " in self.name:
                table = self.name.split(" AS ")[1]  # 有_
            else:
                table = self.name
            if i.split('.')[0] == "".join(table.split("_")):
                self.indices.append(table + i.split('.')[1])  # 可能有问题 没有考虑_缩写的  #考虑的relation的所有列

    def set_selectivity(self, select, selectivity):
        table_name = select.split('.')[0]
        assert table_name == self.name, ValueError(
            'Does not match the table name!!!')
        """
            Save the selection for generate the sql
        """
        self.selectivity[select] = selectivity
        self.embed[self.column_dict[select]][1] = selectivity
        # print(self.name)
        # print(self.embed)

    def set_join(self, column_id):
        self.embed[column_id][2] = 1

    def set_clause(self, selection):  # 存select的条件
        table_name = selection.split(" ")[0].split('.')[0]
        if "(" in table_name:
            table_name = table_name.replace("(", "")
        assert table_name == self.name, ValueError(
            'Does not match the table name!!!')

        self.clauses.append(selection)

    def clause_to_sql(self):
        if len(self.clauses) == 0:
            return self.sql_name
        else:
            sql = '( SELECT * FROM ' + self.sql_name + ' WHERE '
            clauses = ' AND '.join(self.clauses)
            sql += clauses
            sql += ') AS ' + self.name  # abc, abc2
            # self.clauses = [] # 会影响吗
            return sql

    def toSql(self, level):
        if len(self.clauses) == 0:
            sql = ' SELECT * FROM ' + self.sql_name
            return sql
        else:
            sql = ' SELECT * FROM ' + self.sql_name + ' WHERE '
            clauses = ' AND '.join(self.clauses)
            sql += clauses

            # self.clauses = [] # 会影响吗
            return sql


class EmptyQuery(object):
    name = 'EmptyQuery'
    mask = []

    def __init__(self, mask):
        self.mask = mask


class Query(object):
    left = None
    right = None
    name = ''
    join_condition = {}
    join_condition_id = {}
    joined_columns = []
    mask = []
    columns = []
    aliasflag = True

    # aliasflag = False

    def __init__(self, left, right, action_num, col_num):
        global join_conditions, join_conditions_id
        self.joined_columns = []

        self.left = left
        self.right = right
        self.action_num = action_num
        self.allselevtivity = [0 for _ in range(col_num)]
        self.left_selevtivity = [0 for _ in range(col_num)]
        self.right_selevtivity = [0 for _ in range(col_num)]

        for k, v in self.left.embed.items():
            self.allselevtivity[k] = v[1]
            self.left_selevtivity[k] = v[1]

        self.embed = self.left.embed.copy()
        for k, v in self.right.embed.items():
            self.right_selevtivity[k] = v[1]
            if k not in self.embed:
                self.embed[k] = v
                self.allselevtivity[k] = v[1]

        lname = self.left.name
        lname_list = lname.split("&")

        rname = self.right.name
        rname_list = rname.split("&")

        self.name = '&'.join(sorted(lname_list + rname_list))
        self.mask = [x | y for (x, y) in zip(left.mask, right.mask)]

        if self.name in join_conditions:
            self.join_condition = join_conditions[self.name]
            self.join_condition_id = join_conditions_id[self.name]
            for i in self.join_condition:
                self.joined_columns.append(i.split('.')[1])  # 只有列名
        if type(self.left) is Query:
            self.joined_columns = self.joined_columns + self.left.joined_columns

        if type(self.right) is Query:
            self.joined_columns = self.joined_columns + self.right.joined_columns

        self.columns = []
        tmpcolumns = []
        for c in left.columns:
            if " AS " in c:
                self.columns.append(lname + '.' + c.split(' AS ')[1])
                tmpcolumns.append(c.split(' AS ')[1])
            else:
                self.columns.append(lname + '.' + c.split('.')[1])
                tmpcolumns.append(c.split('.')[1])

        for c in right.columns:
            if " AS " in c:
                c = rname + "." + c.split(" AS ")[1]
            if c.split('.')[1] in tmpcolumns:

                new_column = rname + "." + str(c.split('.')[0].split("&")[0] + c.split('.')[0].split("&")[-1]) + "$" + \
                             c.split('.')[1]

                while new_column.split('.')[1] in tmpcolumns:
                    new_column = new_column + "_tmp"
                tmpcolumns.append(new_column.split('.')[1])
                self.columns.append(rname + "." + c.split('.')[1] + " AS " + new_column.split('.')[1])

                for key, val in join_conditions.items():
                    newval = []
                    for v in val:
                        if v == rname + "." + c.split('.')[1]:
                            newval.append(v.replace(rname + "." + c.split('.')[1], new_column))
                            # if "kind_type" in new_column: print(new_column)
                        else:
                            newval.append(v)
                    join_conditions[key] = newval
            else:
                self.columns.append(rname + "." + c.split(".")[1])
                tmpcolumns.append(c.split('.')[1])

        join_conditions, join_conditions_id = self.deleteJoinCondition(lname, rname, join_conditions,
                                                                       join_conditions_id)

        join_conditions, join_conditions_id = self.changeJoinConditions(lname, rname, self.name, join_conditions,
                                                                        join_conditions_id)

    def changeJoinConditions(self, relA, relB, relnew, join_conditions, join_conditions_id):
        conditions = {}
        conditions_id = {}
        relB = relB.split('&')
        relA = relA.split('&')
        for key, value in join_conditions.items():
            if set(relB).issubset(key.split('&')):
                new_key = '&'.join(np.unique(sorted(key.split('&') + relnew.split('&'))))
                value2 = []
                for v in value:
                    if set(relB).issubset(v.split('.')[0].split('&')):
                        value2.append(
                            '&'.join(np.unique(sorted(v.split('.')[0].split('&') + relnew.split('&')))) + '.' +
                            v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]  # 有问题
                else:
                    conditions[new_key] = value2
                    conditions_id[new_key] = join_conditions_id[key]  # 有问题
            elif set(relA).issubset(key.split('&')):
                new_key = '&'.join(np.unique(sorted(key.split('&') + relnew.split('&'))))
                value2 = []
                for v in value:
                    if set(relA).issubset(v.split('.')[0].split('&')):
                        value2.append(
                            '&'.join(np.unique(sorted(v.split('.')[0].split('&') + relnew.split('&')))) + '.' +
                            v.split('.')[1])
                    else:
                        value2.append(v)
                if new_key in conditions:
                    conditions[new_key] = conditions[new_key] + value2
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[new_key] = value2
                    conditions_id[new_key] = join_conditions_id[key]

            else:
                if key in conditions:
                    conditions[key] = conditions[key] + value
                    conditions_id[new_key] = conditions_id[new_key] + join_conditions_id[key]
                else:
                    conditions[key] = value
                    conditions_id[key] = join_conditions_id[key]

        return conditions, conditions_id

    def deleteJoinCondition(self, relA, relB, jc, jc_id):
        conditions = dict(jc)
        conditions_id = dict(jc_id)
        try:
            join_key = '&'.join(np.unique(sorted(relA.split('&') + relB.split('&'))))
            # del conditions[join_key]
            conditions.pop(join_key)
            # del conditions_id[join_key]
            conditions_id.pop(join_key)
        except Exception:
            pass
        return conditions, conditions_id

    def tohint(self):


        if type(self.left) is Relation and type(self.right) is Relation:
            hint = '(' + self.left.name + ' ' + self.right.name + ')'

        elif type(self.left) is Relation and type(self.right) is Query:
            hint = '(' + self.left.name + ' ' + self.right.tohint() + ')'

        elif type(self.left) is Query and type(self.right) is Relation:
            hint = '(' + self.left.tohint() + ' ' + self.right.name + ')'

        elif type(self.left) is Query and type(self.right) is Query:
            hint = '(' +  self.left.tohint() + ' ' + self.right.tohint() + ')'

        return hint


    def to_tree_structure(self, num):
        tree = Tree()
        tree.num = num
        tree.all_selectivity = torch.tensor(self.allselevtivity)
        tree.left_all_selectivity = torch.tensor(self.left_selevtivity)
        tree.right_all_selectivity = torch.tensor(self.right_selevtivity)

        # tree.index = index
        if type(self.left) is Relation and type(self.right) is Relation:
            tree.l_name = self.left.name
            tree.r_name = self.right.name

            tree.l_table_id = self.left.id
            tree.r_table_id = self.right.id

            tree.l_table_embed = [self.left.embed[value] for key, value in enumerate(self.left.embed)]
            tree.r_table_embed = [self.right.embed[value] for key, value in enumerate(self.right.embed)]
            tree.joined_num = 1
            tree.hint = '(' + self.left.name + ' ' + self.right.name + ')'

        elif type(self.left) is Relation and type(self.right) is Query:
            tree.l_name = self.left.name
            tree.l_table_id = self.left.id
            tree.l_table_embed = [self.left.embed[value] for key, value in enumerate(self.left.embed)]

            tree.r_name = self.right.to_tree_structure(tree.num)
            # tree.r_name = self.right.to_tree_structure(2 * index + 2)
            tree.joined_num = 1 + tree.r_name.joined_num
            tree.hint = '(' + self.left.name + ' ' + tree.r_name.hint + ')'
            #tree.hint = '(' + tree.r_name.hint + ' ' + self.left.name + ')'
        elif type(self.left) is Query and type(self.right) is Relation:
            tree.l_name = self.left.to_tree_structure(tree.num)
            # tree.l_name = self.left.to_tree_structure(2 * index + 1)
            tree.r_name = self.right.name
            tree.r_table_id = self.right.id
            tree.r_table_embed = [self.right.embed[value] for key, value in enumerate(self.right.embed)]
            tree.joined_num = 1 + tree.l_name.joined_num
            tree.hint = '(' + tree.l_name.hint + ' ' + self.right.name + ')'

        elif type(self.left) is Query and type(self.right) is Query:
            tree.l_name = self.left.to_tree_structure(tree.num)
            tree.r_name = self.right.to_tree_structure(tree.num)
            # tree.l_name = self.left.to_tree_structure(2 * index + 1)
            # tree.r_name = self.right.to_tree_structure(2 * index + 2)
            tree.joined_num = tree.r_name.joined_num + tree.l_name.joined_num + 1
            tree.hint = '(' + tree.l_name.hint + ' ' + tree.r_name.hint + ')'

        else:
            raise ValueError("Not supported (Sub)Query!")

        # self.join_condition_id

        if len(self.join_condition) is not 0:
            if self.join_condition_id[0] in self.left.embed and self.join_condition_id[1] in self.right.embed:
                tree.l_column_embed.append(self.left.embed[self.join_condition_id[0]])
                tree.r_column_embed.append(self.right.embed[self.join_condition_id[1]])
                #tree.l_column.append(self.join_condition[0])
                #tree.r_column.append(self.join_condition[1])
                tree.l_column_id.append(self.join_condition_id[0])
                tree.r_column_id.append(self.join_condition_id[1])

            elif self.join_condition_id[1] in self.left.embed and self.join_condition_id[0] in self.right.embed:
                tree.l_column_embed.append(self.left.embed[self.join_condition_id[1]])
                tree.r_column_embed.append(self.right.embed[self.join_condition_id[0]])
                #tree.l_column.append(self.join_condition[1])
                #tree.r_column.append(self.join_condition[0])
                tree.l_column_id.append(self.join_condition_id[1])
                tree.r_column_id.append(self.join_condition_id[0])


            if len(self.join_condition) > 2:
                for i in range(2, len(self.join_condition), 2):

                    if self.join_condition_id[i] in self.left.embed and self.join_condition_id[i + 1] in self.right.embed:
                        tree.l_column_embed.append(self.left.embed[self.join_condition_id[i]])
                        tree.r_column_embed.append(self.right.embed[self.join_condition_id[i+1]])
                        #tree.l_column.append(self.join_condition[i])
                        #tree.r_column.append(self.join_condition[i+1])
                        tree.l_column_id.append(self.join_condition_id[i])
                        tree.r_column_id.append(self.join_condition_id[i+1])

                    elif self.join_condition_id[1] in self.left.embed and self.join_condition_id[0] in self.right.embed:
                        tree.l_column_embed.append(self.left.embed[self.join_condition_id[i+1]])
                        tree.r_column_embed.append(self.right.embed[self.join_condition_id[i]])
                        #tree.l_column.append(self.join_condition[i+1])
                        #tree.r_column.append(self.join_condition[i])
                        tree.l_column_id.append(self.join_condition_id[i+1])
                        tree.r_column_id.append(self.join_condition_id[i])


        else:
            print(self.join_condition)

            raise ValueError("No Candidate Join Conditions!")

        return tree


class Rel_Columns(object):
    name = ''
    id = None
    selectivity = None

    def __init__(self, name, id, selectivity):
        self.name = name
        self.id = id
        self.selectivity = selectivity
        self.table = name.spilt('.')[0]


def getJoinConditions():
    return join_conditions
