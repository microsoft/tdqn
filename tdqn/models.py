import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class TDQN(nn.Module):
    def __init__(self, args, template_size, vocab_size, vocab_size_act):
        super(TDQN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size_act, args.embedding_size)

        self.state_network = StateNetwork(args, vocab_size)

        self.t_scorer = nn.Linear(args.hidden_size, template_size)
        self.o1_scorer = nn.Linear(args.hidden_size, vocab_size_act)
        self.o2_scorer = nn.Linear(args.hidden_size, vocab_size_act)
        self.args = args
        self.template_size = template_size
        self.vocab_size_act = vocab_size_act

    def forward(self, state):
        x, h = self.state_network(state)

        q_t = self.t_scorer(x)
        q_o1 = self.o1_scorer(x)
        q_o2 = self.o2_scorer(x)

        return q_t, q_o1, q_o2

    def act(self, state, epsilon):
        with torch.no_grad():
            state = torch.LongTensor(state).unsqueeze(0).permute(1, 0, 2).cuda()
            q_t, q_o1, q_o2 = self.forward(state)
            t, o1, o2 = F.softmax(q_t, dim=1).multinomial(num_samples=1).item(),\
                        F.softmax(q_o1, dim=1).multinomial(num_samples=1).item(),\
                        F.softmax(q_o2, dim=1).multinomial(num_samples=1).item()
            q_t = q_t[0,t].item()
            q_o1 = q_o1[0,o1].item()
            q_o2 = q_o2[0,o2].item()
            return t, o1, o2, q_t, q_o1, q_o2


    def poly_act(self, state, n_samples=512, replacement=True):
        ''' Samples many times from the model, optionally with replacement. '''
        with torch.no_grad():
            state = torch.LongTensor(state).unsqueeze(0).permute(1, 0, 2).cuda()
            q_t, q_o1, q_o2 = self.forward(state)
            t, o1, o2 = F.softmax(q_t, dim=1).multinomial(n_samples, replacement)[0],\
                        F.softmax(q_o1, dim=1).multinomial(n_samples, replacement)[0],\
                        F.softmax(q_o2, dim=1).multinomial(n_samples, replacement)[0]
            qv_t = torch.index_select(q_t, 1, t).squeeze().cpu().detach().numpy()
            qv_o1 = torch.index_select(q_o1, 1, o1).squeeze().cpu().detach().numpy()
            qv_o2 = torch.index_select(q_o2, 1, o2).squeeze().cpu().detach().numpy()
            return t.cpu().numpy(), o1.cpu().numpy(), o2.cpu().numpy(), qv_t, qv_o1, qv_o2

    def flatten_parameters(self):
        self.state_network.flatten_parameters()



class StateNetwork(nn.Module):
    def __init__(self, args, vocab_size):
        super(StateNetwork, self).__init__()
        self.args = args

        self.enc_look = PackedEncoderRNN(vocab_size, args.hidden_size)
        self.enc_inv = PackedEncoderRNN(vocab_size, args.hidden_size)
        self.enc_ob = PackedEncoderRNN(vocab_size, args.hidden_size)
        self.enc_preva = PackedEncoderRNN(vocab_size, args.hidden_size)

        self.fcx = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.fch = nn.Linear(args.hidden_size * 4, args.hidden_size)

    def forward(self, obs):
        x_l, h_l = self.enc_look(obs[0, :, :], self.enc_look.initHidden(self.args.batch_size))
        x_i, h_i = self.enc_inv(obs[1, :, :], self.enc_inv.initHidden(self.args.batch_size))
        x_o, h_o = self.enc_ob(obs[2, :, :], self.enc_ob.initHidden(self.args.batch_size))
        x_p, h_p = self.enc_preva(obs[3, :, :], self.enc_preva.initHidden(self.args.batch_size))

        x = F.relu(self.fcx(torch.cat((x_l, x_i, x_o, x_p), dim=1)))
        h = F.relu(self.fch(torch.cat((h_l, h_i, h_o, h_p), dim=2)))

        return x, h

    def flatten_parameters(self):
        self.enc_look.flatten_parameters()
        self.enc_inv.flatten_parameters()
        self.enc_ob.flatten_parameters()
        self.enc_preva.flatten_parameters()


class PackedEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PackedEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input).permute(1,0,2) # T x Batch x EmbDim
        if hidden is None:
            hidden = self.initHidden(input.size(0))

        # Pack the padded batch of sequences
        lengths = torch.tensor([torch.nonzero(n)[-1] + 1 for n in input], dtype=torch.long).cuda()

        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        output, hidden = self.gru(packed, hidden)
        # Unpack the padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # Return only the last timestep of output for each sequence
        idx = (lengths-1).view(-1, 1).expand(len(lengths), output.size(2)).unsqueeze(0)
        output = output.gather(0, idx).squeeze(0)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()

    def flatten_parameters(self):
        self.gru.flatten_parameters()
