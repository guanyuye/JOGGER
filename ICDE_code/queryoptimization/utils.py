import logging
import psycopg2

def setup_seed(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def init_pg():
    try:
        conn = psycopg2.connect(
            database='im_database', user='imdb', password='', host='127.0.0.1', port='5432')

    except:
        print("I am unable to connect to the database")
    cursor = conn.cursor()
    return cursor
