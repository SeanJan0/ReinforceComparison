import time
import neat
import numpy as np

class EnvEvaluator:
    def __init__(self, test_env, activator, disp_env=None, max_env_steps=None, min_env_score=None, checkpoint_freq=5, checkpoint_prefix=None, threads=None):
        self.env = test_env
        self.disp_env = disp_env
        self.activator = activator
        self.max_env_steps = max_env_steps
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_freq = checkpoint_freq
        self.min_env_score = min_env_score
        self.threads = threads
        
    def eval_genome(self, genome, config):
        """Evaluates the input Genome"""
        state  = self.env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        step_num = 0
        done = False
        fitness = 0
        while True:
            # print("{},{}".format(step_num,fitness))
            step_num+=1
            #if done exit this genome
            if done:
                break   
            #if reached maximum number of steps exit this genome
            elif self.max_env_steps is not None and step_num >= self.max_env_steps:
                break
            #if reached 
            elif self.min_env_score is not None and fitness <= self.min_env_score:
                break
            #Otherwise fetch the next action based upon the current state and conduct another step
            else:
                action = self.activator(net, state)
                state, reward, done, _ = self.env.step(action)
                fitness += reward
        return fitness
    
    def disp_genome(self, genome, config):
        """Displays how the input Genome plays"""
        if self.disp_env is not None:
            state  = self.disp_env.reset()
        else:
            state  = self.env.reset()
            
        if self.disp_env is None:
            self.env.render()
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
            elif self.max_env_steps is not None and step_num >= self.max_env_steps:
                break
            #if reached 
            elif self.min_env_score is not None and fitness <= self.min_env_score:
                break
            #Otherwise fetch the next action based upon the current state and conduct another step
            else:
                action = self.activator(net, state)
                if self.disp_env is not None:
                    state, reward, done, _ = self.disp_env.step(action)
                else:  
                    state, reward, done, _ = self.env.step(action)
                fitness += reward
            if self.disp_env is None:
                self.env.render()
        return fitness

    def eval_many_genome(self, genomes, config):
        """evaluates the performance every given genome"""
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
        if self.threads is None or self.threads == 1:
            winner = p.run(self.eval_many_genome, num_generations)
        else:
            print("USING THREADED EVALUATOR")
            pe = neat.ThreadedEvaluator(self.threads,self.eval_genome)
            winner = p.run(pe.evaluate, num_generations)
        

        # Display the winning genome.
        # print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        win_fit = self.disp_genome(winner, config)
        print("\nWinner Fitness:{}".format(win_fit))