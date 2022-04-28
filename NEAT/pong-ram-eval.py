from matplotlib.pyplot import gray
import numpy as np
import os
import gym
from EnvEvaluator import EnvEvaluator

#General Config Parameters
generations = 200 #Number of Generations to run
max_env_steps = 20000 #Maximum number of steps, None=Inf (Often Regulated within the simulation)
min_score = None #Minimum score the AI can achieve before the system resets
train_env = gym.make("ALE/Pong-ram-v5", full_action_space=False)#, render_mode='human') #Gym Environemnt
disp_env = gym.make("ALE/Pong-ram-v5", full_action_space=False, render_mode='human') #Gym Environemnt for displaying gameplay
ram_indexes = [51,49,54] #player_y, ball_y, ball_x

def activate(net, states):
    """Evaluates the input net at the given state for the Pong-V5 Environment"""
    output = np.argmax(np.array(net.activate(states)))
    return output

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pong-ram.cfg')
    prefix = 'NEAT/checkpoints/Pong-ram-v5/neat-checkpoint-'
    evaluator = EnvEvaluator(train_env, activate, 
                             disp_env=disp_env, 
                             max_env_steps=max_env_steps, 
                             min_env_score=min_score,
                             ram_indicies = ram_indexes,
                             threads=10)
    checkpoint = 'NEAT/checkpoints/Pong-ram-v5/neat-checkpoint-1999'
    evaluator.eval_checkpoint_genome(config_path, checkpoint)