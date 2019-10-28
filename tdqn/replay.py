from collections import deque
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PriorityReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priority_buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, valid_acts):
        if reward > 0:
            self.priority_buffer.append((state, action, reward, next_state, done, valid_acts))
        else:
            self.buffer.append((state, action, reward, next_state, done, valid_acts))

    def sample(self, batch_size, rho):
        pbatch = int(batch_size * rho)
        batch = int(batch_size * (1 - rho))
        if pbatch > len(self.priority_buffer):
            pbatch = len(self.priority_buffer)
            batch = batch_size - len(self.priority_buffer)
        elif batch > len(self.buffer):
            batch = len(self.buffer)
            pbatch = batch_size - len(self.buffer)

        if pbatch == 0:
            state, action, reward, next_state, done, valid = zip(*random.sample(self.buffer, batch))

            return list(state), action, reward, list(next_state), done, valid
        if batch == 0:
            pstate, paction, preward, pnext_state, pdone, pvalid = zip(*random.sample(self.priority_buffer, pbatch))
            return list(pstate), paction, preward, list(pnext_state), pdone, pvalid

        state, action, reward, next_state, done, valid = zip(*random.sample(self.buffer, batch))
        pstate, paction, preward, pnext_state, pdone, pvalid = zip(*random.sample(self.priority_buffer, pbatch))

        return pstate + state, paction + action, preward + reward, pnext_state + next_state, pdone + done, pvalid + valid

    def __len__(self):
        return len(self.buffer)
