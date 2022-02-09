import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import scipy.special as sp
import math
import random
import numpy as np
from ..run.model import DQN,Net1,Net2, FLOAT_MAX, FLOAT_MIN,DQ
from torch.optim.lr_scheduler import StepLR

Experience=namedtuple('Experience',('old_obs', 'action_num', 'obs', 'cost', 'done'))

steps_done=0

class ReplayMemory(object):
    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)
    def push (self,prio,*args):
        self.memory.append(Experience(*args))
    def sample (self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)


class PriorityReplayMemory(ReplayMemory):
    def __init__(self,capacity):
        super().__init__(capacity=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.join = deque([], maxlen=capacity)

    def push (self,prio,*args):
        self.memory.append(Experience(*args))
        self.priorities.append(prio)
        self.join.append(prio)
        #print(self.priorities)

    def sample (self,batch_size, priority_scale=1.0):
        sample_size = batch_size
        sample_probs = self.get_probabilities(priority_scale)
        # print("sample_probs",sample_probs)
        # print("len(sample_probs)",len(sample_probs))
        # print("range(self.memory)",range(len(self.memory)))
        # print("len(self.memory)", len(self.memory))
        # print("sample_size",sample_size)
        memory_pool = [ i for i in range(len(self.memory))]
        sample_indices = random.choices(memory_pool, k=sample_size, weights=sample_probs)
        batch = []
        for i in range(len(sample_indices)):
            batch.append(self.memory[sample_indices[i]])
        #print(len(batch))
        return batch,sample_indices

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def set_priorities(self, indices, errors, offset=0):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e)*self.join[i] + offset



class agent(object):
    def __init__(self,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE, num_col, num_rel, num_actions,memory_size):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        #self.in_put_size=in_put_size
        #self.out_put_size=out_put_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Experience = namedtuple('Experience',('old_obs', 'action_num', 'obs', 'cost', 'done'))
        #self.model_list = model_list[DQN, Net1]
        #self.idx = 1
        #model_list[1]()
        self.model_state='train'

        self.policy_net = DQ(num_col, num_rel, num_actions).to(self.device)
        self.target_net = DQ(num_col, num_rel, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory=ReplayMemory(memory_size)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def select_action(self,obs,is_train):

        global steps_done
        if is_train =='train':
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * steps_done / self.EPS_DECAY)
            steps_done += 1

            if sample > eps_threshold:  #将0换成 eps_threshold
                with torch.no_grad():
                    probs = self.policy_net(obs).cpu()
                    # action_mask = obs["action_mask"]
                    # action_mask = np.clip(np.log(action_mask), -np.inf, np.inf)
                    # #+ action_mask
                    final_probs = sp.softmax(probs)
                    action_num = np.argmax(final_probs)

            else:
                action_total = len(obs["action_mask"])
                candidate_action = np.arange(0, action_total)
                probs = np.ones(action_total, dtype=np.float)
                action_mask = obs["action_mask"]
                action_mask = np.clip(np.log(action_mask), -np.inf, np.inf)
                final_probs = sp.softmax(probs + action_mask)
                action_num = int(np.random.choice(a=candidate_action, size=1, p=final_probs))


        elif is_train =='eval':
            with torch.no_grad():
                probs = self.policy_net(obs).cpu()
                # action_mask = obs["action_mask"]
                # action_mask = np.clip(np.log(action_mask), -np.inf, np.inf)
                # #+ action_mask
                final_probs = sp.softmax(probs)
                action_num = np.argmax(final_probs)

        return action_num


    def optimize_model(self):
        print(len(self.memory) )

        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Experience(*zip(*transitions))

        non_final_mask = tuple(map(lambda s: s is not None,
                  batch.obs))


        non_final_mask = non_final_mask
        print(non_final_mask)

        non_final_next_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.obs
                                           if s is not None], dim=0)

        non_final_next_states = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.obs
                                           if s is not None], dim=0)

        next_tree =  [s["tree"] for s in batch.obs if s is not None]


        #print(batch.obs)

        next_link_mtx =   torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
                                           if s is not None and non_final_mask.tolist()[key] is not False ], dim=0)  #是少的


        next_table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
             if s is not None and non_final_mask.tolist()[key] is not False]  # 是少的

        next_states_dict={}
        next_states_dict["db"]= non_final_next_states
        next_states_dict["action_mask"] = non_final_next_states_mask
        next_states_dict["tree"] = next_tree
        next_states_dict["link_mtx"] = next_link_mtx
        next_states_dict["all_table_embed"] = next_table_embed



        non_final_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.old_obs
                                                if s is not None], dim=0)

        state_batch = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.old_obs
                                 if s is not None], dim=0)

        tree = [s["tree"] for s in batch.old_obs]

        link_mtx = torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
             if s is not None ], dim=0)  # 是少的

        table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
             if s is not None ] # 是少的


        states_dict = {}
        states_dict["db"] = state_batch
        states_dict["action_mask"] = non_final_states_mask
        states_dict["tree"] = tree
        states_dict["link_mtx"] = link_mtx
        states_dict["all_table_embed"] = table_embed

        action_batch = torch.cat(batch.action_num).unsqueeze(1)

        reward_batch = torch.cat(batch.cost)

        state_action_values = self.policy_net(states_dict).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)

        next_state_values[non_final_mask] = self.target_net( next_states_dict).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def Floss(value,targetvalue):
    with torch.no_grad():
        disl1 = (torch.abs(value-targetvalue)<0.15).float()
    with torch.no_grad():
        disl2 = 1-((value>targetvalue).float()*(targetvalue>1-0.1).float())  #4
    return torch.mean(disl2*((1-disl1)*torch.abs(value-targetvalue)*(targetvalue+1)+disl1*(value-targetvalue)*(value-targetvalue)))

class BasePolicy(object):
    def __init__(self, config, model:nn.Module):
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']
        self.EPS_START = config['EPS_START']
        self.EPS_END = config['EPS_END']
        self.EPS_DECAY = config['EPS_DECAY']

        self.device = torch.device(config['device'])
        self.prioritized = config['prioritized']

        self.model_state = 'train'
        self.policy_net = model(config['num_col'], config['num_rel'], config['num_actions'], config).to(self.device)
        self.target_net = model(config['num_col'], config['num_rel'], config['num_actions'], config).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()



        if self.prioritized :
            self.memory = PriorityReplayMemory(config['CAPACITY'])

        else:
            self.memory = ReplayMemory(config['CAPACITY'])

        if config['optim'] == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay']) #

        elif config['optim'] == 'rms':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config['learning_rate'],weight_decay=config['weight_decay'])

        else:
            raise ValueError('Not supported Loss')

        #self.scheduler = StepLR(self.optimizer, step_size=5000, gamma=0.95)

        if config['loss'] == 'l1':
            self.criterion = nn.SmoothL1Loss()
        elif config['loss'] == 'huber':
            self.criterion = nn.HuberLoss()
        elif config['loss'] == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = None
            #raise ValueError('Not supported Loss')

        self.num_actions = config['num_actions']
        
    
    def select_action(self, obs, is_train=True):
        global steps_done
        if is_train:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * steps_done / self.EPS_DECAY)
            steps_done += 1
            #print("eps_threshold",eps_threshold)
            if random.random() > eps_threshold:
                with torch.no_grad():
                    action_num = self.policy_net(obs).max(0)[1]
                    #action_num = self.policy_net(obs).min(0)[1]
                    #print("self.policy_net(obs)",self.policy_net(obs).max(0))
            else:
                #print("random")
                probs = torch.ones(self.num_actions).to(self.device)
                action_mask = torch.tensor(obs["action_mask"]).to(self.device)
                action_mask = torch.clip(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
                probs = torch.softmax(probs + action_mask, dim=-1)
                action_num = torch.multinomial(probs, 1)
        else:
            with torch.no_grad():
                action_num = self.policy_net(obs).max(0)[1]
                #action_num = self.policy_net(obs).min(0)[1]
        return int(action_num)
    
    def optimize_model(self):
        #print("optimize_model")
        #namedtuple('Experience',('old_obs', 'action_num', 'obs', 'cost', 'done'))
        if len(self.memory) < self.BATCH_SIZE:
            return 99999

        if self.prioritized:
            transitions, sample_indices = self.memory.sample(self.BATCH_SIZE)
        else:
            transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Experience(*zip(*transitions))

        non_final_mask = tuple(map(lambda s: s is not None,
                                                batch.obs))


        # if True not in non_final_mask:
        #     #print("all_the_last_state")
        #     next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        # else :
        #     non_final_mask = torch.tensor(non_final_mask, device=self.device, dtype=torch.bool)
        #     # non_final_next_states_mask = [torch.FloatTensor([s["action_mask"]]) for key, s in enumerate(batch.obs)
        #     #                  if s is not None and non_final_mask.tolist()[key] is not False]
        #     non_final_next_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for  s in batch.obs
        #          if s is not None ], dim=0)
        #     '''
        #
        #     non_final_next_states = torch.cat([s["db"].unsqueeze(0) for  s in batch.obs
        #                                        if s is not None], dim=0)
        #     '''
        #     # non_final_next_states = torch.cat([torch.FloatTensor([s["db"]]) for key, s in enumerate(batch.obs)
        #     #      if s is not None and non_final_mask.tolist()[key] is not False], dim=0)
        #
        #     next_tree = [s["tree"] for s in batch.obs
        #                  if s is not None ]
        #     # next_tree = [s["tree"] for key, s in enumerate(batch.obs)
        #     #      if s is not None and non_final_mask.tolist()[key] is not False]
        #
        #     # print(batch.obs)
        #
        #
        #     next_link_mtx = torch.stack(
        #         [s["link_mtx"] for key, s in enumerate(batch.old_obs)
        #          if non_final_mask.tolist()[key] is not False], dim=0)  #
        #
        #     next_table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
        #         if non_final_mask.tolist()[key] is not False]  # 是少的
        #
        #     next_states_dict = {}
        #     '''
        #     next_states_dict["db"] = non_final_next_states
        #     '''
        #     next_states_dict["action_mask"] = non_final_next_states_mask
        #     next_states_dict["tree"] = next_tree
        #
        #     next_states_dict["link_mtx"] = next_link_mtx
        #     next_states_dict["all_table_embed"] = next_table_embed
        #
        #     next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        #     #next_state_values[non_final_mask] = self.target_net(next_states_dict).max(1)[0].detach()
        #
        #     #next_state_values[non_final_mask] = self.target_net(next_states_dict).min(1)[0].detach()
        #     # print(self.policy_net(next_states_dict).shape)
        #     # print(self.policy_net(next_states_dict).argmax(1))
        #     next_state_index_in_policy  = self.policy_net(next_states_dict).argmax(1).unsqueeze(1).detach()
        #     next_state_values[non_final_mask] = self.target_net(next_states_dict).gather(1, next_state_index_in_policy).squeeze(1).detach()



        states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.old_obs
                                               ], dim=0)
        #non_final_states_mask = torch.cat([torch.FloatTensor([s["action_mask"]]) for s in batch.old_obs
        #                                        if s is not None], dim=0)

        state_batch = torch.cat([s["db"].unsqueeze(0) for s in batch.old_obs
                                 ], dim=0)
        #print(state_batch.shape)


        # state_batch = torch.cat([torch.FloatTensor([s["db"]]) for s in batch.old_obs
        #                          if s is not None], dim=0)
        tree = [s["tree"] for s in batch.old_obs]


        link_mtx = torch.stack(
            [s["link_mtx"] for key, s in enumerate(batch.old_obs)
              ], dim=0)  # 是少的
        table_embed = [s["all_table_embed"] for key, s in enumerate(batch.old_obs)
              ] # 是少的

        states_dict = {}

        states_dict["db"] = state_batch
        states_dict["all_table_embed"] = table_embed

        #print("states_dict[db]", type(states_dict["db"]))
        states_dict["action_mask"] = states_mask
        states_dict["tree"] = tree
        states_dict["link_mtx"] = link_mtx


        action_batch = torch.cat(batch.action_num).unsqueeze(1)
        reward_batch = torch.cat(batch.cost)
        #print("reward_batch", reward_batch.shape)
        state_action_values = self.policy_net(states_dict).gather(1, action_batch)
        '''
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch
        '''
        expected_state_action_values = reward_batch

        if self.prioritized :
            error = expected_state_action_values.unsqueeze(1)-state_action_values
            error2=error.squeeze().detach().clone().cpu().tolist()
            self.memory.set_priorities(sample_indices,error2)

        if self.criterion is not None:
            loss = self.criterion(state_action_values,
                                  expected_state_action_values.unsqueeze(1))
        else:
            loss = Floss(state_action_values,
                                  expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            # if  param.requires_grad == True:
            #     param.grad.data.clamp_(-1, 1)
            torch.nn.utils.clip_grad_norm_(param, 10, norm_type=2)

        self.optimizer.step()
        #self.scheduler.step()
        #print(loss)

        return loss.item()