import numpy as np
import os
import gym
from EnvEvaluator import EnvEvaluator

#General Config Parameters
generations = 250 #Number of Generations to run
max_env_steps = None #Maximum number of steps, None=Inf
env = gym.make('CartPole-v1') #Gym Environemnt

def activate(net, states):
    """Evaluates the input net at the given state for the Cartpole-V1 Environment"""
    output = round(np.array(net.activate(states))[0])
    return output

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'cartpole.cfg')
    print(config_path)
    prefix = 'NEAT/checkpoints/CartPole-V1/neat-checkpoint-'
    evaluator = EnvEvaluator(env, activate, checkpoint_prefix=prefix)
    evaluator.run(config_path, generations)