import time
import math, random
import numpy as np
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import logger
import copy

from replay import *
from schedule import *
from models import TDQN

from env import *
import jericho
from jericho.template_action_generator import TemplateActionGenerator

import sentencepiece as spm


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log



class TDQN_Trainer(object):
    def __init__(self, args):
        configure_logger(args.output_dir)
        log(args)
        self.args = args

        self.log_freq = args.log_freq
        self.update_freq = args.update_freq_td
        self.update_freq_tar = args.update_freq_tar
        self.filename = 'tdqn'

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.binding = jericho.load_bindings(args.rom_path)
        self.vocab_act, self.vocab_act_rev = self.load_vocab_act(args.rom_path)
        vocab_size = len(self.sp)
        vocab_size_act = len(self.vocab_act.keys())

        self.template_generator = TemplateActionGenerator(self.binding)
        self.template_size = len(self.template_generator.templates)

        if args.replay_buffer_type == 'priority':
            self.replay_buffer = PriorityReplayBuffer(int(args.replay_buffer_size))
        elif args.replay_buffer_type == 'standard':
            self.replay_buffer = ReplayBuffer(int(args.replay_buffer_size))

        self.model = TDQN(args, self.template_size, vocab_size, vocab_size_act).cuda()
        self.target_model = TDQN(args, self.template_size, vocab_size, vocab_size_act).cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.num_steps = args.steps
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.rho = args.rho

        self.bce_loss = nn.BCELoss()

    def load_vocab_act(self, rom_path):
        #loading vocab directly from Jericho
        env = FrotzEnv(rom_path)
        vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
        vocab[0] = ' '
        vocab[1] = '<s>'
        vocab_rev = {v: idx for idx, v in vocab.items()}
        env.close()
        return vocab, vocab_rev

    def state_rep_generator(self, state_description):
        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            state_description = state_description.replace(rm, '')

        state_description = state_description.split('|')

        ret = [self.sp.encode_as_ids('<s>' + s_desc + '</s>') for s_desc in state_description]

        return pad_sequences(ret, maxlen=self.args.max_seq_len)

    def plot(self, frame_idx, rewards, losses, completion_steps):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('frame %s. steps: %s' % (frame_idx, np.mean(completion_steps[-10:])))
        plt.plot(completion_steps)
        plt.subplot(133)
        plt.title('loss-lstm-dqn')
        plt.plot(losses)
        # txt = "Gamma:" + str(self.gamma) + ", Num Frames:" + str(self.num_frames) + ", E Decay:" + str(epsilon_decay)
        plt.figtext(0.5, 0.01, self.filename, wrap=True, horizontalalignment='center', fontsize=12)
        # plt.show()
        fig.savefig('plots/' + self.filename + '_' + str(frame_idx) + '.png')

    def compute_td_loss(self):
        state, action, reward, next_state, done, valid = self.replay_buffer.sample(self.batch_size, self.rho)
        action = torch.LongTensor(action).cuda()
        state = torch.LongTensor(state).permute(1, 0, 2).cuda()
        next_state = torch.LongTensor(next_state).permute(1, 0, 2).detach().cuda()
        template_targets = torch.stack([v[0] for v in valid]).cuda()
        obj_targets = torch.stack([v[1] for v in valid]).cuda()

        decode_steps = []
        for t in action[:, 0]:
            decode_steps.append(self.template_generator.templates[t.item()].count('OBJ'))

        template = action[:, 0]
        object1 = action[:, 1]
        object2 = action[:, 2]
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(1 * done).cuda()

        o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
        for d, st in enumerate(decode_steps):
            if st > 1:
                o1_mask[d] = 1
                o2_mask[d] = 1
            elif st == 1:
                o1_mask[d] = 1

        o1_mask, o2_mask = torch.FloatTensor(o1_mask).cuda(), torch.FloatTensor(o2_mask).cuda()

        self.model.flatten_parameters()
        q_t, q_o1, q_o2 = self.model(state)

        supervised_loss = self.bce_loss(F.softmax(q_t, dim=1), template_targets)+\
                          self.bce_loss(F.softmax(q_o1, dim=1), obj_targets)+\
                          self.bce_loss(F.softmax(q_o2, dim=1), obj_targets)
        tb.logkv_mean('SupervisedLoss', supervised_loss.item())

        self.target_model.flatten_parameters()
        next_q_t, next_q_o1, next_q_o2 = self.target_model(next_state)

        q_t = q_t.gather(1, template.unsqueeze(1)).squeeze(1)
        q_o1 = q_o1.gather(1, object1.unsqueeze(1)).squeeze(1)
        q_o2 = q_o2.gather(1, object2.unsqueeze(1)).squeeze(1)

        next_q_t = next_q_t.max(1)[0]
        next_q_o1 = next_q_o1.max(1)[0]
        next_q_o2 = next_q_o2.max(1)[0]

        td_loss = F.smooth_l1_loss(q_t, (reward + self.gamma * next_q_t).detach()) +\
                  F.smooth_l1_loss(q_o1 * o1_mask, o1_mask * (reward + self.gamma * next_q_o1).detach()) +\
                  F.smooth_l1_loss(q_o2 * o2_mask, o2_mask * (reward + self.gamma * next_q_o2).detach())

        tb.logkv_mean('TDLoss', td_loss.item())
        loss = td_loss + supervised_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss

    def tmpl_to_str(self, template_idx, o1_id, o2_id):
        template_str = self.template_generator.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', self.vocab_act[o1_id])
        else:
            return template_str.replace('OBJ', self.vocab_act[o1_id], 1)\
                               .replace('OBJ', self.vocab_act[o2_id], 1)


    def generate_targets_multilabel(self, valid_acts):
        template_targets = torch.zeros([self.template_size])
        obj_targets = torch.zeros([len(self.vocab_act.keys())])
        for act in valid_acts:
            template_targets[act.template_id] = 1
            for obj_id in act.obj_ids:
                obj_targets[obj_id] = 1
        return template_targets, obj_targets


    def train(self):
        start = time.time()
        env = JerichoEnv(self.args.rom_path, 0, self.vocab_act_rev,
                         self.args.env_step_limit)
        env.create()

        episode = 1
        state_text, info = env.reset()
        state_rep = self.state_rep_generator(state_text)

        for frame_idx in range(1, self.num_steps + 1):
            found_valid_action = False
            while not found_valid_action:
                templates, o1s, o2s, q_ts, q_o1s, q_o2s = self.model.poly_act(state_rep)
                for template, o1, o2, q_t, q_o1, q_o2 in zip(templates, o1s, o2s, q_ts, q_o1s, q_o2s):
                    action = [template, o1, o2]
                    action_str = self.tmpl_to_str(template, o1, o2)
                    next_state_text, reward, done, info = env.step(action_str)
                    if info['action_valid'] == True:
                        found_valid_action = True
                        break

            if episode % 100 == 0:
                log('Action: {} Q_t: {:.2f} Q_o1: {:.2f} Q_o2: {:.2f}'.format(action_str, q_t, q_o1, q_o2))
                log('Obs: {}'.format(clean(next_state_text.split('|')[2])))
                log('Reward {}: {}'.format(env.steps, reward))

            valid_acts = info['valid']
            template_targets, obj_targets = self.generate_targets_multilabel(valid_acts)
            next_state_rep = self.state_rep_generator(next_state_text)
            self.replay_buffer.push(state_rep, action, reward, next_state_rep,
                                    done, (template_targets, obj_targets))
            state_text = next_state_text
            state_rep = next_state_rep

            if done:
                score = info['score']
                if episode % 100 == 0:
                    log('Episode {} Score {}\n'.format(episode, score))
                tb.logkv_mean('EpisodeScore', score)
                state_text, info = env.reset()
                state_rep = self.state_rep_generator(state_text)
                episode += 1

            if len(self.replay_buffer) > self.batch_size:
                if frame_idx % self.update_freq == 0:
                    loss = self.compute_td_loss()
                    tb.logkv_mean('Loss', loss.item())

            if frame_idx % self.update_freq_tar == 0:
                self.target_model = copy.deepcopy(self.model)

            if frame_idx % self.log_freq == 0:
                tb.logkv('Step', frame_idx)
                tb.logkv('FPS', int(frame_idx/(time.time()-start)))
                tb.dumpkvs()

        env.close()

        parameters = {
            'model': self.model,
            'target': self.target_model,
            'replay_buffer': self.replay_buffer
        }
        torch.save(parameters, pjoin(self.args.output_dir, self.filename + '_final.pt'))


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x
