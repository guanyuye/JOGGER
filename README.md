Requirements
Python 3.7
Pytorch 1.7
deepwalk 1.0.3


# ICDE2022_JOGGER
efficient Join Order selection learninG with Graph-basEd Representation (JOGGER)

# Important parameters
Here we have listed the most important parameters you need to configure to run RTOS on a new database. 

- schemafile
    - <a href ="https://github.com/gregrahn/join-order-benchmark/blob/master/schema.sql"> a sample</a>
- sytheticDir
    - Directory contains the sytheic workload(queries), more sythetic queries will help RTOS make better plans. 
    - It is nesscery when you want apply RTOS for a new database.  
- JOBDir
    - Directory contains all JOB queries. 
- Configure of PostgreSQL
    - dbName : the name of database 
    - userName : the user name of database
    - password : your password of userName
    - ip : ip address of PostgreSQL
    - port : port of PostgreSQL

# Requirements
- Pytorch 1.0
- Python 3.7
- psqlparse

# Run the JOB   
1. Download JOB dataset from https://github.com/gregrahn/join-order-benchmark
2. Add JOB queries in the Directory: ICDE2022_JOGGER/ICDE_code/agents/queries/crossval_sens/IMDB_data.txt
3. Run encode_table.py to build the adjacent matrix to reflect the primary-foreign key relationships
4. Generates the table embedding matrix according to the adjacent matrix by the deepwalk package of Python 
5. Run the train_JOGGER_main.py to optimize the model
