from os.path import basename
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import *
from jericho.defines import *
import redis

def load_vocab_rev(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: idx for idx, v in vocab.items()}
    return vocab_rev


class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''
    def __init__(self, rom_path, seed, step_limit=None):
        self.rom_path = rom_path
        self.bindings = load_bindings(rom_path)
        self.act_gen = TemplateActionGenerator(self.bindings)
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.env = None
        self.conn = None
        self.vocab_rev = None

    def create(self):
        self.env = FrotzEnv(self.rom_path, self.seed)
        self.vocab_rev = load_vocab_rev(self.env)
        self.conn = redis.Redis(host='localhost', port=6379, db=0)
        self.conn.flushdb()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait','yes','no']
        if not done:
            try:
                save = self.env.save_str()
                look, _, _, _ = self.env.step('look')
                info['look'] = look
                self.env.load_str(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv
                self.env.load_str(save)
                # Get the valid actions for this state
                world_state_hash = self.env.get_world_state_hash()
                valid = self.conn.get(world_state_hash)
                if valid is None:
                    objs = [o[0] for o in self.env.identify_interactive_objects(ob)]
                    obj_ids = [self.vocab_rev[o[:self.bindings['max_word_length']]] for o in objs]
                    acts = self.act_gen.generate_template_actions(objs, obj_ids)
                    valid = self.env.find_valid_actions(acts)
                    redis_valid_value = '/'.join([str(a) for a in valid])
                    self.conn.set(world_state_hash, redis_valid_value)
                    valid = [a.action for a in valid]
                else:
                    valid = valid.decode('cp1252')
                    if valid:
                        valid = [eval(a).action for a in valid.split('/')]
                    else:
                        valid = []
                if len(valid) == 0:
                    valid = ['wait','yes','no']
                info['valid'] = valid
            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean(ob), done, info))
        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        save = self.env.save_str()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.load_str(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.load_str(save)
        objs = [o[0] for o in self.env.identify_interactive_objects(initial_ob)]
        acts = self.act_gen.generate_actions(objs)
        valid = self.env.find_valid_actions(acts)
        info['valid'] = valid
        self.steps = 0
        return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def close(self):
        self.env.close()
