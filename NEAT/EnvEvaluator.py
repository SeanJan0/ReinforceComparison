import time
import neat
import numpy as np

class EnvEvaluator:
    def __init__(self, env, activator, max_env_steps=None, checkpoint_freq=5, checkpoint_prefix=None):
        self.env = env
        self.activator = activator
        self.max_env_steps = max_env_steps
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_freq = checkpoint_freq
        
    def eval_genome(self, genome, config, demo=False):
        """Evaluates the input Genome"""
        state  = self.env.reset()
        # self.env.render()
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
            if self.max_env_steps is not None and step_num >= self.max_env_steps:
                break
            #Otherwise fetch the next action based upon the current state and conduct another step
            else:
                action = self.activator(net, state)
                state, reward, done, _ = self.env.step(action)
                fitness += reward
            if demo:
                self.env.render()
        return fitness

    def eval_many_genome(self, genomes, config):
        """evalueates the performance every given genome"""
        fitnesses = np.zeros(np.shape(genomes)[0])
        for genome_id, genome in genomes:
            fitness = self.eval_genome(genome, config)
            genome.fitness = fitness
                

    def run(self, config_file, num_generations):
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
        if self.checkpoint_prefix is None:
            p.add_reporter(neat.Checkpointer(self.checkpoint_freq))
        else:
            p.add_reporter(neat.Checkpointer(self.checkpoint_freq,filename_prefix=self.checkpoint_prefix))
            

        # Run for up to the given number of generations.
        winner = p.run(self.eval_many_genome, num_generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        win_fit = self.eval_genome(winner, config, demo=True)
        print("\nWinner Fitness:{}".format(win_fit))