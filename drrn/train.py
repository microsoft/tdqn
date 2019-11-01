import subprocess
import time
import os
import torch
import logger
import argparse
import yaml
import jericho
from os.path import basename, dirname
from drrn import DRRN_Agent
from vec_env import VecEnv
from env import JerichoEnv
from jericho.util import clean


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('tensorboard', log_dir),
                                 logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log


def evaluate(agent, env, nb_episodes=1):
    with torch.no_grad():
        total_score = 0
        for ep in range(nb_episodes):
            log("Starting evaluation episode {}".format(ep))
            score = evaluate_episode(agent, env)
            log("Evaluation episode {} ended with score {}\n\n".format(ep, score))
            total_score += score
        avg_score = total_score / nb_episodes
        return avg_score


def evaluate_episode(agent, env):
    step = 0
    done = False
    ob, info = env.reset()
    state = agent.build_state([ob], [info])[0]
    log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
    while not done:
        valid_acts = info['valid']
        valid_ids = agent.encode(valid_acts)
        _, action_idx, action_values = agent.act([state], [valid_ids], sample=False)
        action_idx = action_idx[0]
        action_values = action_values[0]
        action_str = valid_acts[action_idx]
        log('Action{}: {}, Q-Value {:.2f}'.format(step, action_str, action_values[action_idx].item()))
        s = ''
        for idx, (act, val) in enumerate(sorted(zip(valid_acts, action_values), key=lambda x: x[1], reverse=True), 1):
            s += "{}){:.2f} {} ".format(idx, val.item(), act)
        log('Q-Values: {}'.format(s))
        ob, rew, done, info = env.step(action_str)
        log("Reward{}: {}, Score {}, Done {}".format(step, rew, info['score'], done))
        step += 1
        log('Obs{}: {} Inv: {} Desc: {}'.format(step, clean(ob), clean(info['inv']), clean(info['look'])))
        state = agent.build_state([ob], [info])[0]
    return info['score']


def train(agent, eval_env, envs, max_steps, update_freq, eval_freq, checkpoint_freq, log_freq):
    start = time.time()
    obs, infos = envs.reset()
    states = agent.build_state(obs, infos)
    valid_ids = [agent.encode(info['valid']) for info in infos]
    for step in range(1, max_steps+1):
        action_ids, action_idxs, _ = agent.act(states, valid_ids)
        action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]
        obs, rewards, dones, infos = envs.step(action_strs)
        for done, info in zip(dones, infos):
            if done:
                tb.logkv_mean('EpisodeScore', info['score'])
        next_states = agent.build_state(obs, infos)
        next_valids = [agent.encode(info['valid']) for info in infos]
        for state, act, rew, next_state, valids, done in \
            zip(states, action_ids, rewards, next_states, next_valids, dones):
            agent.observe(state, act, rew, next_state, valids, done)
        states = next_states
        valid_ids = next_valids
        if step % log_freq == 0:
            tb.logkv('Step', step)
            tb.logkv("FPS", int((step*envs.num_envs)/(time.time()-start)))
            tb.dumpkvs()
        if step % update_freq == 0:
            loss = agent.update()
            if loss is not None:
                tb.logkv_mean('Loss', loss)
        if step % checkpoint_freq == 0:
            agent.save()
        if step % eval_freq == 0:
            eval_score = evaluate(agent, eval_env)
            tb.logkv('EvalScore', eval_score)
            tb.dumpkvs()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_envs', default=8, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--checkpoint_freq', default=5000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--memory_size', default=500000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--clip', default=5, type=float)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    return parser.parse_args()


def start_redis():
    print('Starting Redis')
    subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no'])
    time.sleep(1)

def main():
    assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    print(args)
    configure_logger(args.output_dir)
    start_redis()
    agent = DRRN_Agent(args)
    env = JerichoEnv(args.rom_path, args.seed, args.env_step_limit)
    envs = VecEnv(args.num_envs, env)
    env.create() # Create the environment for evaluation
    train(agent, env, envs, args.max_steps, args.update_freq, args.eval_freq,
          args.checkpoint_freq, args.log_freq)


def interactive_run(env):
    ob, info = env.reset()
    while True:
        print(clean(ob), 'Reward', reward, 'Done', done, 'Valid', info)
        ob, reward, done, info = env.step(input())


if __name__ == "__main__":
    main()
