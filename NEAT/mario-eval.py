import numpy as np
import os
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from EnvEvaluator import EnvEvaluator

#General Config Parameters
generations = 1000 #Number of Generations to run
max_env_steps = None #Maximum number of steps, None=Inf (Often Regulated within the simulation)
min_score = None 
im_size = (15,16)
train_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3") #Gym Environemnt
train_env = JoypadSpace(train_env, SIMPLE_MOVEMENT)
train_env.seed(1)
# disp_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0") #Gym Environemnt
# disp_env = JoypadSpace(disp_env, SIMPLE_MOVEMENT)
# disp_env.seed(1)

def activate(net, states):
    """Evaluates the input net at the given state for the MountainCar-V0 Environment"""
    output = np.argmax(np.array(net.activate(states)))
    return output

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'mario.cfg')
    print(config_path)
    prefix = 'NEAT/checkpoints/Mario/neat-checkpoint-'
    evaluator = EnvEvaluator(train_env, activate, 
                            #  disp_env=disp_env,
                             max_env_steps=max_env_steps, 
                             min_env_score=min_score,
                             checkpoint_prefix=prefix, 
                             threads=None,
                             flat=True,
                             resize_image=im_size,
                             gray_scale=True,
                             append_info=True,
                             disp_train=False)       
    checkpoint ='NEAT/checkpoints/Mario3/neat-checkpoint-403'
    evaluator.eval_checkpoint_genome(config_path, checkpoint, id=37440)