from agents.run.masking_env_tree import CrossVal
import queryoptimization.utils as utils
from agents.run.model import DQN, Net1, Net2 ,DQ,RTOS,Net3
from agents.run.agents import BasePolicy, agent
from config_files import  POLICY_CONFIG, MODEL_CONFIG
import scipy.special as sp
import argparse
import torch
import numpy as np
import os
import random
import time
import psycopg2
model_list = [Net1,Net2,DQ,RTOS,Net3,DQN]
policy_list = [BasePolicy]

def main(config, log_dir):

    env = CrossVal({'fold_idx':config['fold_idx'], 'process':0})
    eval_env = CrossVal({'fold_idx':config['fold_idx'], 'process':1})
    device = config['device']
    obs,table_num = env.reset()
    #print(obs['db'])

    action_num = env.action_space.n
    model = model_list[config['model']]
    model_name= str(model).split(".")[-1].split("'")[0]

    config['num_col'] = env.wrapped.num_of_columns
    config['num_rel'] = env.wrapped.num_of_relations
    config['num_actions'] = env.action_space.n

    agent_a = policy_list[config['policy']](config, model)
    first_ep = True
    loss_0 = 0.0
    epi = 0
    use_best_join=0
    best_order = {}
    train_order = []
    best_order_value={}
    sql_num = env.wrapped.sql_id
    priority_list=[]
    old_obs_list=[]
    action_num_list=[]
    obs_list=[]
    cost_list=[]
    done_list=[]
    each_state_best_value={}
    old_time = time.time()
    for i in range(config['episodes']):

        if first_ep is True:
            first_ep = False
            obs['tree'] = None
            obs['link_mtx']=env.wrapped.query.link_mtx.to(device).to(torch.float32)
            obs['all_table_embed'] = env.wrapped.table_embeds

        else:
            obs['link_mtx'] = old_obs['link_mtx']
            obs['all_table_embed'] = old_obs['all_table_embed'].copy()

        if  use_best_join < 0.7 or sql_num not in best_order:
            action_num = agent_a.select_action(obs, True)
            train_order.append(action_num)
        else:
            action_num = temp_best_order[best_action_index]
            best_action_index +=1


        old_obs = obs.copy()
        obs, cost, done, sub_trees,join_num = env.step(action_num)
        cost = torch.tensor([cost]).to(device)
        action_num = torch.tensor([action_num]).to(device)

        if config['Normalization'] == 'sum':
            w1 = join_num/np.sum(np.array([i for i in range(1,table_num)]))
            np.arange(1,table_num)
            w2 = table_num/103

        elif config['Normalization'] == 'softmax':

            w1 = torch.tensor([i for i in range(1,table_num)])
            w1 = sp.softmax(w1)[join_num-1].item()
            w2 = torch.tensor([0,0,0,0,4,5,6,7,8,9,10,11,12,0,14,0,0,17])
            w2 = sp.softmax(w2)[table_num].item()

        priority = (1 + config['Lambda1']*w1) * (1 + config['Lambda2']*w2)

        if not done:
            obs['tree'] = sub_trees
            #agent_a.memory.push(priority, old_obs, action_num, obs, cost, done)     #old_obs: mask,db , tree ,link_mtx,all_table_embed
                                                                              #    obs: mask,db , tree
            priority_list.append(priority)
            old_obs_list.append(old_obs)
            action_num_list.append(action_num)

            obs_list.append(obs)
            cost_list.append(cost)
            done_list.append(done)

        else:

            obs = None
            priority_list.append(priority)
            old_obs_list.append(old_obs)
            action_num_list.append(action_num)
            obs_list.append(obs)
            cost_list.append(cost)
            done_list.append(done)
            #agent_a.memory.push(priority, old_obs, action_num, obs, cost, done)

            temp_cost = cost.item()
            if use_best_join > 0.7:
                print("add_best_cost", temp_cost)

            if sql_num in best_order:
                if temp_cost > best_order_value[sql_num]:
                    print("now_best",temp_cost)
                    print(best_order_value[sql_num])
                    best_order[sql_num]=train_order
                    best_order_value[sql_num] = temp_cost
            else:
                best_order[sql_num] = train_order
                best_order_value[sql_num] = temp_cost #sql的最佳reward
            train_order = []

            for pe in range(len(priority_list)):
                state_index= str(sql_num)+str(old_obs_list[pe]['db'])+str(action_num_list[pe].item())
                if state_index not in each_state_best_value:
                    each_state_best_value[state_index] = temp_cost
                    agent_a.memory.push(priority_list[pe], old_obs_list[pe], action_num_list[pe], obs_list[pe], cost,
                                        done_list[pe])
                elif each_state_best_value[state_index] < temp_cost:
                    each_state_best_value[state_index] = temp_cost
                    agent_a.memory.push(priority_list[pe], old_obs_list[pe], action_num_list[pe], obs_list[pe], cost,
                                        done_list[pe])
                else:
                    agent_a.memory.push(priority_list[pe], old_obs_list[pe], action_num_list[pe], obs_list[pe], torch.tensor([each_state_best_value[state_index]]).to(device),
                                        done_list[pe])
            priority_list = []
            old_obs_list = []
            action_num_list = []
            obs_list = []
            cost_list = []
            done_list = []
            loss0 = agent_a.optimize_model()
            loss_0 += loss0

        if done is True:
            if epi % config['TARGET_UPDATE'] == 0:
                agent_a.target_net.load_state_dict(agent_a.policy_net.state_dict())
            if epi % config['Test'] ==0:
                loss_0 = 0.0
                with torch.no_grad():
                    avg_rows, avg_cost = eval(agent_a, epi, eval_env, device, log_dir)
                logger.info('Episode {}/{}, Avg Rows {}, Avg Cost {}'.format(i, epi, avg_rows, avg_cost))
            if  epi % 200 ==0:
                PATH = r'/data/ygy/code_list/join_mod/save_model_dict/' + str(epi) + '_' + str(
                    config['BATCH_SIZE']) + '_' + str(model_name)+ str(policy_name) + '_para.pth'
                torch.save({'policy_net_state_dict': agent_a.policy_net.state_dict(),
                            'optimizer': agent_a.optimizer.state_dict()}, PATH)
            obs, table_num = env.reset()
            first_ep = True
            epi += 1
            sql_num=  env.wrapped.sql_id
            use_best_join = random.random()
            if use_best_join > 0.7:
                if sql_num in best_order:
                    temp_best_order = best_order[sql_num]
                    best_action_index= 0




def eval(agent_a, episodes, env, device, log_dir):
    test_num = env.wrapped.num_test
    estimate_rows = []
    estimate_cost = []
    for idx in range(test_num):
        obs,table_num = env.reset()
        obs['tree'] = None
        done = False
        while done is False:

            obs['link_mtx'] = env.wrapped.query.link_mtx.to(device).to(torch.float32)
            obs['all_table_embed'] = env.wrapped.table_embeds
            action_num = agent_a.select_action(obs, False)
            obs, cost, done, sub_trees,join_num = env.step(action_num)
            obs['tree'] = sub_trees

        if done is True:
            sql = env.wrapped.sql
            sql_id = env.wrapped.sql_id
            estimatedRows, estimatedcosts = get_cost_rows(sql)
            estimate_rows.append(float(estimatedRows))
            estimate_cost.append(float(estimatedcosts))
            file_name = os.path.join(log_dir, 'test_episodes_{}.txt'.format(episodes))
            with open(file_name, mode='a') as f:
                f.write(sql_id+'|'+sql+'estimatedRows:'+estimatedRows+ '|' + 'estimatedcosts:'+estimatedcosts+'\n')

    return sum(estimate_rows)/len(estimate_rows), sum(estimate_cost)/len(estimate_cost)
            
def get_cost_rows(sql):
    #print(sql)
    cursor = utils.init_pg()
    cursor.execute(""" EXPLAIN """ + sql)
    #print(sql)
    rows = cursor.fetchall()

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedRows = row0[1].replace("rows=", "")

    row0 = rows[0][0].split("(cost=")[1].split(' ')
    estimatedcosts = row0[0].split("..")[1]
    return estimatedRows, estimatedcosts

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Initialize Parameters!')
    parser.add_argument('-b', '--BATCH_SIZE', default=256, type=int, help='BATCH_SIZE for training')
    parser.add_argument('-ed', '--emb_dim', default=128, type=int)
    parser.add_argument('-gd', '--graph_dim', default=128, type=int)
    parser.add_argument('-e', '--episodes', default=100000000, type=int)
    parser.add_argument('-vis', '--CUDA_VISIBLE_DEVICES', default='1', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--log_path', default='logs', type=str)
    parser.add_argument('--model', default=3, type=int)  # [Net1,Net2,DQ,RTOS,Net3,DQN]
    parser.add_argument('--policy', default=0, type=int)
    parser.add_argument('-s', '--SEED', default=10, type=int)
    parser.add_argument('-p', '--policy_name', default='DQN', type=str)
    parser.add_argument('-f', '--fold_idx', default=5, type=int, help='The index of folds for cross validation experiments')
    parser.add_argument('-lr', '--learning_rate', default=3e-3, type=float)
    parser.add_argument('-ga', '--GAMMA', default=0.999, type=float)
    parser.add_argument('-wd', '--weight_decay', default=0.005, type=float)

    config = vars(parser.parse_args())
    config = dict(**config, **POLICY_CONFIG, **MODEL_CONFIG)
    model_name = model_list[config['model']].__name__
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']
    policy_name = config['policy_name']  # "DQN" # can modify here

    import os, datetime
    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])

    utils.setup_seed(config["SEED"])

    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((model_name, policy_name, timestamp))
    log_dir = os.path.join(config['log_path'], loglogs)
    os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, "running_log")
    logger = utils.get_logger(log_file_name)
    logger.info(config)
    main(config, log_dir)

