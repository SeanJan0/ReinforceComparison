import numpy as np
import os
import gym
from EnvEvaluator import EnvEvaluator

#General Config Parameters
generations = 1000 #Number of Generations to run
max_env_steps = 1000 #Maximum number of steps, None=Inf (Often Regulated within the simulation)
min_score = None #Minimum score the AI can achieve before the system resets
im_size = (32,32)
train_env = gym.make("BreakoutNoFrameskip-v4")#, render_mode='human') #Gym Environemnt
disp_env = gym.make("BreakoutNoFrameskip-v4", render_mode='human') #Gym Environemnt for displaying gameplay

def activate(net, states):
    """Evaluates the input net at the given state for the Pong-V5 Environment"""
    output = np.argmax(np.array(net.activate(states)))
    return output

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    # print(train_env.action_space)
    config_path = os.path.join(local_dir, 'breakout.cfg')
    prefix = 'NEAT/checkpoints/breakout/neat-checkpoint-'
    evaluator = EnvEvaluator(train_env, activate, 
                             disp_env=disp_env, 
                             max_env_steps=max_env_steps, 
                             min_env_score=min_score,
                             checkpoint_prefix=prefix,
                             gray_scale=True, 
                             resize_image=im_size,
                             flat=True,
                             threads=None)
    evaluator.run(config_path, generations)