import subprocess
import time
import redis
from os.path import basename, dirname
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator
from jericho.util import *
from jericho.defines import *

def start_redis():
    print('Starting Redis')
    subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no'])
    time.sleep(1)


class JerichoEnv:
    def __init__(self, rom_path, seed, vocab_rev, step_limit=None):
        self.rom_path = rom_path
        self.bindings = load_bindings(rom_path)
        self.act_gen = TemplateActionGenerator(self.bindings)
        self.seed = seed
        self.steps = 0
        self.step_limit = step_limit
        self.vocab_rev = vocab_rev
        self.env = None
        self.conn = None

    def create(self):
        self.env = FrotzEnv(self.rom_path, self.seed)
        start_redis()
        self.conn = redis.Redis(host='localhost', port=6379, db=0)
        self.conn.flushdb()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        action_valid = done or self.env.world_changed()
        info['action_valid'] = action_valid
        if not action_valid: # Exit early for invalid actions
            return None, None, None, info
        if action_valid:
            self.steps += 1
        # Initialize with default values
        look = 'unknown'
        inv = 'unknown'
        info['valid'] = []
        if not done:
            try:
                save = self.env.save_str()
                look, _, _, _ = self.env.step('look')
                self.env.load_str(save)
                inv, _, _, _ = self.env.step('inventory')
                self.env.load_str(save)
                # Find Valid actions
                world_state_hash = self.env.get_world_state_hash()
                valid = self.conn.get(world_state_hash)
                if valid is None:
                    objs = [o[0] for o in self.env.identify_interactive_objects(ob)]
                    obj_ids = [self.vocab_rev[o[:self.bindings['max_word_length']]] for o in objs]
                    acts = self.act_gen.generate_template_actions(objs, obj_ids)
                    valid = self.env.find_valid_actions(acts)
                    redis_valid_value = '/'.join([str(a) for a in valid])
                    self.conn.set(world_state_hash, redis_valid_value)
                else:
                    valid = valid.decode('cp1252')
                    if valid:
                        valid = [eval(a) for a in valid.split('/')]
                info['valid'] = valid
            except RuntimeError:
                print('RuntimeError: {}, Done: {}, Info: {}'.format(clean(ob), done, info))
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        ob = look + '|' + inv + '|' + ob + '|' + action
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()
        try:
            save = self.env.save_str()
            look, _, _, _ = self.env.step('look')
            self.env.load_str(save)
            inv, _, _, _ = self.env.step('inventory')
            self.env.load_str(save)
        except RuntimeError:
            print('RuntimeError: {}, Info: {}'.format(initial_ob, info))
            look, inv = ''
        self.steps = 0
        initial_ob = look + '|' + inv + '|' + initial_ob + '|' + 'look'
        return initial_ob, info

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def close(self):
        self.env.close()
        self.conn.shutdown(save=True)
