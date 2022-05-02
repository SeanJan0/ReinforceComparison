import numpy as np
import os
import gym
from EnvEvaluator import EnvEvaluator

#General Config Parameters
generations = 250  #Number of Generations to run
max_env_steps = None #Maximum number of steps, None=Inf (Often Regulated within the simulation)
env = gym.make('MountainCar-v0') #Gym Environemnt

def activate(net, states):
    """Evaluates the input net at the given state for the MountainCar-V0 Environment"""
    output = np.argmax(np.array(net.activate(states)))
    return output

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'mountaincar.cfg')
    print(config_path)
    prefix = 'NEAT/checkpoints/MountainCar-V0/neat-checkpoint-'
    evaluator = EnvEvaluator(env, activate, checkpoint_prefix=prefix)
    evaluator.run(config_path, generations)