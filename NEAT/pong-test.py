import numpy as np
import os
import gym
from EnvEvaluator import EnvEvaluator

#General Config Parameters
generations = 10 #Number of Generations to run
max_env_steps = None #Maximum number of steps, None=Inf (Often Regulated within the simulation)
min_score = -10 #Minimum score the AI can achieve before the system resets
train_env = gym.make("ALE/Pong-ram-v5") #Gym Environemnt
disp_env = gym.make("ALE/Pong-ram-v5", render_mode='human') #Gym Environemnt for displaying gameplay

def activate(net, states):
    """Evaluates the input net at the given state for the Pong-V5 Environment"""
    output = np.argmax(np.array(net.activate(states)))
    if output == 0:
        return 2 ## up
    elif output == 1:
        return 3
    else:
        print("THIS ISNT MEANT TO HAPPEN")
        exit(1)
    # return output

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'pong.cfg')
    prefix = 'NEAT/checkpoints/Pong-v5/neat-checkpoint-'
    evaluator = EnvEvaluator(train_env, activate, disp_env=disp_env, max_env_steps=max_env_steps, min_env_score=min_score,checkpoint_prefix=prefix, threads=10)
    evaluator.run(config_path, generations)