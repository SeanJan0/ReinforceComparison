import numpy as np
import os
import neat
import gym

#General Config Parameters
generations = 100 #Number of Generations to run
max_env_steps = None #Maximum number of steps, None=Inf
env = gym.make('CartPole-v1') #Gym Environemnt

def activate(net, states):
    """Evaluates the input net at the given state"""
    output = np.array(net.activate(states))
    output[output>0.5]=int(1)
    output[output<=0.5]=int(0)
    if np.size(output)>1:
        return output
    else:
        return int(output[0])

def eval_genome(genome,config):
    """Evaluates the input Genome"""
    state  = env.reset()
    env.render()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    step_num = 0
    done = False
    fitness = 0
    while True:
        step_num+=1
        #if done exit this genome
        if done:
            break   
        #if reached maximum number of steps exit this genome
        if max_env_steps is not None and step_num >= max_env_steps:
            break
        #Otherwise fetch the next action based upon the current state and conduct another step
        else:
            action = activate(net, state)
            state, reward, done, _ = env.step(action)
            fitness += reward
    return fitness

def eval_many_genome(genomes, config):
    """evalueates the performance every given genome"""
    fitnesses = np.zeros(np.shape(genomes)[0])
    for genome_id, genome in genomes:
        fitness = eval_genome(genome, config)
        genome.fitness = fitness
            

def run(config_file, num_generations):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_many_genome, num_generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    win_fit = eval_genome(winner, config)
    print("\nWinner Fitness:{}".format(win_fit))

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_many_genome, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'cartpole.cfg')
    print(config_path)
    run(config_path,generations)