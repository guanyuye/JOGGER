
# ICDE2022_JOGGER
efficient Join Order selection learninG with Graph-basEd Representation (JOGGER) is an efficient optimizer for solving the Join order Selection(JOS) problem. It utilizes the curriculum learning, reinforcement learning and a tailored-tree-based attention module to generate query plan.  

# Requirements
- Python 3.7 
- Pytorch 1.7
- psqlparse
- deepwalk 1.0.3

# Run the JOB   
1. Download JOB dataset from https://github.com/gregrahn/join-order-benchmark
2. Add JOB queries in the Directory: ICDE2022_JOGGER/ICDE_code/agents/queries/crossval_sens/IMDB_data.txt
3. Run `encode_table.py` to build the adjacent matrix to reflect the primary-foreign key relationships
4. Generate the table embedding matrix according to the adjacent matrix by the deepwalk package of Python 
5. Run `train_JOGGER_main.py` to optimize the model

# Run the TPC-H   
1. Download TPC-H from http://www.tpc.org/tpc_documents_current_versions/current_specifications.asp 
2. Generate TPC-H queries from 22 templates
3. Add TPC-H queries in the Directory: ICDE2022_JOGGER/ICDE_code/agents/queries/crossval_sens/TPCH_data.txt
4. Run `encode_table.py` to build the adjacent matrix to reflect the primary-foreign key relationships
5. Generate the table embedding matrix according to the adjacent matrix by the deepwalk package of Python 
6. Run `train_JOGGER_main.py` to optimize the model
