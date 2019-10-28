import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin
from memory import ReplayMemory, Transition, State
from model import DRRN
from util import *
import logger
import sentencepiece as spm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DRRN_Agent:
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.network = DRRN(len(self.sp), args.embedding_dim, args.hidden_dim).to(device)
        self.memory = ReplayMemory(args.memory_size)
        self.save_path = args.output_dir
        self.clip = args.clip
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)


    def observe(self, state, act, rew, next_state, next_acts, done):
        self.memory.push(state, act, rew, next_state, next_acts, done)


    def build_state(self, obs, infos):
        """ Returns a state representation built from various info sources. """
        obs_ids = [self.sp.EncodeAsIds(o) for o in obs]
        look_ids = [self.sp.EncodeAsIds(info['look']) for info in infos]
        inv_ids = [self.sp.EncodeAsIds(info['inv']) for info in infos]
        return [State(ob, lk, inv) for ob, lk, inv in zip(obs_ids, look_ids, inv_ids)]


    def encode(self, obs_list):
        """ Encode a list of observations """
        return [self.sp.EncodeAsIds(o) for o in obs_list]


    def act(self, states, poss_acts, sample=True):
        """ Returns a string action from poss_acts. """
        idxs, values = self.network.act(states, poss_acts, sample)
        act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
        return act_ids, idxs, values


    def update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute Q(s', a') for all a'
        # TODO: Use a target network???
        next_qvals = self.network(batch.next_state, batch.next_acts)
        # Take the max over next q-values
        next_qvals = torch.tensor([vals.max() for vals in next_qvals], device=device)
        # Zero all the next_qvals that are done
        next_qvals = next_qvals * (1-torch.tensor(batch.done, dtype=torch.float, device=device))
        targets = torch.tensor(batch.reward, dtype=torch.float, device=device) + self.gamma * next_qvals

        # Next compute Q(s, a)
        # Nest each action in a list - so that it becomes the only admissible cmd
        nested_acts = tuple([[a] for a in batch.act])
        qvals = self.network(batch.state, nested_acts)
        # Combine the qvals: Maybe just do a greedy max for generality
        qvals = torch.cat(qvals)

        # Compute Huber loss
        loss = F.smooth_l1_loss(qvals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()


    def load(self):
        try:
            self.memory = pickle.load(open(pjoin(self.save_path, 'memory.pkl'), 'rb'))
            self.network = torch.load(pjoin(self.save_path, 'model.pt'))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())


    def save(self):
        try:
            pickle.dump(self.memory, open(pjoin(self.save_path, 'memory.pkl'), 'wb'))
            torch.save(self.network, pjoin(self.save_path, 'model.pt'))
        except Exception as e:
            print("Error saving model.")
            logging.error(traceback.format_exc())
