import numpy as np
import torch
join_meta_file = '/data/ygy/code_list/join_mod/total_graph/joinable_tables.txt'
table_set = set()
join_edge_list = []
with open(join_meta_file, 'r') as f:
    for line in f.readlines():
        tables = line.split('\n')[0].split(',')
        for t in tables:
            if t not in table_set:
                table_set.add(t)
        join_edge_list.append(tables)

# print(table_set)
table_set = sorted(table_set)
# print(table_set)
table_id = dict(zip(table_set, range(len(table_set))))
# print(table_id)
print(join_edge_list)
join_edge_list = sorted(join_edge_list)

# join_out_file = '/data/ygy/code_list/join_mod/total_graph/joinable_tables_id.txt'
# with open(join_out_file, 'w') as fn:
#     for pair in join_edge_list:
#         t0 = table_id[pair[0]]
#         t1 = table_id[pair[1]]
#         line = '{} {}\n'.format(t0, t1) if t0 < t1 else '{} {}\n'.format(t1, t0)
#         fn.write(line)

# join_out_file = '/data/ygy/code_list/join_mod/total_graph/out_deepwalk.embeddings'
# with open(join_out_file, 'r') as fn:
#     for line in fn.readlines():
#         embed= line.strip().split(" ")
#         for i in range (len(embed)):
#             if i==0:
#                 id= int(embed[i])
#             else :
#                 embed[i]=float(embed[i])
#         id =embed[0]
#         embed = torch.tensor(embed[1:])
#         print(id)
#         print(embed)

