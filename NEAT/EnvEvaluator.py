import re
import time
from cv2 import resize
from matplotlib.pyplot import gray
import neat
from neat.six_util import iteritems
import cv2
import numpy as np
import random


class EnvEvaluator:
    def __init__(self, test_env, activator,
                 disp_env=None,
                 max_env_steps=None,
                 min_env_score=None,
                 checkpoint_freq=5,
                 checkpoint_prefix=None,
                 ram_indicies=None,
                 threads=None,
                 flat=False,
                 resize_image=None,
                 gray_scale=False,
                 append_info=False,
                 disp_train=False):
        self.env = test_env
        self.disp_env = disp_env
        self.activator = activator
        self.max_env_steps = max_env_steps
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_freq = checkpoint_freq
        self.min_env_score = min_env_score
        self.ram_indicies = ram_indicies
        self.threads = threads
        self.resize_image = resize_image
        self.flatten = flat
        self.gray = gray_scale
        self.append_info = append_info
        self.disp_train = disp_train

    def eval_genome(self, genome, config):
        """Evaluates the input Genome"""
        state = self.env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        step_num = 0
        done = False
        info = dict()
        info = {'coins': 0,
                'flag_get': False,
                'life': 1,
                'score': 0,
                'time': 0,
                'x_pos': 0,
                'y_pos': 0}
        fitness = 0
        old = 0
        while True:
            step_num += 1
            # Isolate the imortant info from the ram if provided
            if self.ram_indicies is not None:
                state = state[self.ram_indicies]
            # if done exit this genome
            if done:
                break
            # if reached maximum number of steps exit this genome
            elif self.max_env_steps is not None and step_num >= self.max_env_steps:
                break
            # if reached
            elif self.min_env_score is not None and fitness <= self.min_env_score:
                break
            # Otherwise fetch the next action based upon the current state and conduct another step
            else:
                # gray scale the image if needed
                if self.gray:
                    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                # Down Scale or Upscale Image as specified
                if self.resize_image is not None:
                    # cv2.imshow("Before-Scale",state)
                    # , interpolation=cv2.INTER_CUBIC)
                    state = cv2.resize(state, self.resize_image)
                    # cv2.imshow("After-Scale",state)
                    # cv2.waitKey(0)
                # Flatten State if needed
                if self.flatten:
                    state = state.flatten()
                if self.append_info and info is not None:
                    for key in info.keys():
                        if key in ['status', 'stage', 'world', 'x_pos_screen']:
                            pass
                        else:
                            state = np.append(state, info[key])
                action = self.activator(net, state)
                state, reward, done, info = self.env.step(action)
                fitness += reward
                if step_num % 250 == 0:
                    if info is not None:
                        if 'x_pos' in info.keys():
                            if old == info['x_pos']:
                                break
                            else:
                                old = info['x_pos']
            if self.disp_train:
                self.env.render()
        # print(step_num)
        return fitness

    def disp_genome(self, genome, config):
        """Displays how the input Genome plays"""
        while True:
            state = self.env.reset()
            if self.disp_env is None:
                self.env.render()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            step_num = 0
            done = False
            info = dict()
            info = {'coins': 0,
                    'flag_get': False,
                    'life': 1,
                    'score': 0,
                    'time': 0,
                    'x_pos': 0,
                    'y_pos': 0}
            fitness = 0
            old = 0
            actions=[]
            while True:
                step_num += 1
                # Isolate the imortant info from the ram if provided
                if self.ram_indicies is not None:
                    state = state[self.ram_indicies]
                # if done exit this genome
                if done:
                    break
                # if reached maximum number of steps exit this genome
                elif self.max_env_steps is not None and step_num >= self.max_env_steps:
                    break
                # if reached
                elif self.min_env_score is not None and fitness <= self.min_env_score:
                    break
                # Otherwise fetch the next action based upon the current state and conduct another step
                else:
                    # gray scale the image if needed
                    if self.gray:
                        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
                    # Down Scale or Upscale Image as specified
                    if self.resize_image is not None:
                        # cv2.imshow("Before-Scale",state)
                        # , interpolation=cv2.INTER_CUBIC)
                        state = cv2.resize(state, self.resize_image)
                        # cv2.imshow("After-Scale",state)
                        # cv2.waitKey(0)
                    # Flatten State if needed
                    if self.flatten:
                        state = state.flatten()
                    if self.append_info and info is not None:
                        for key in info.keys():
                            if key in ['status', 'stage', 'world', 'x_pos_screen']:
                                pass
                            else:
                                state = np.append(state, info[key])
                    action = int(self.activator(net, state))
                    actions = np.append(actions, [action])
                    state, reward, done, info = self.env.step(action)
                    fitness += reward
                    if step_num % 250 == 0:
                        if info is not None:
                            if 'x_pos' in info.keys():
                                if old == info['x_pos']:
                                    break
                                else:
                                    old = info['x_pos']
                    if self.disp_env is None:
                        self.env.render()
            fitness2 = 0
            if self.disp_env is not None:
                state = self.disp_env.reset()
                for action in actions:
                    # self.disp_env.render()
                    state, reward, done, info = self.disp_env.step(int(action))
                    fitness2+=reward
                    if done:
                        print("DONE")
                        break
            repeat = input("Press Enter to See again, or X to exit")
            if repeat == 'X':
                print("DISP FITNESS: {}".format(fitness2))
                break
        return fitness

    def eval_many_genome(self, genomes, config):
        """evaluates the performance every given genome"""
        fitnesses = np.zeros(np.shape(genomes)[0])
        i = 0
        for genome_id, genome in genomes:
            # print("{}%".format(round(i/len(genomes)*100, 2)), end='\r')
            fitness = self.eval_genome(genome, config)
            # print(type(fitness))
            genome.fitness = int(fitness)
            i += 1

    def eval_checkpoint_genome(self, config_file, name, id=None):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Checkpointer.restore_checkpoint(name)
        winner = None
        if id==None:
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            winner = p.run(self.eval_many_genome, 1)
        else:
            genomes = list(iteritems(p.population))
            for genome_id,genome in genomes:
                if genome_id==id:
                    winner = genome
                    break
            if winner is None:
                print("WE SHOULD NOT BE HERE")
        # Display the winning genome.
        # print('\nBest genome:\n{!s}'.format(winner))
        input("Training Finished Press Enter to see performance...")
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        win_fit = self.disp_genome(winner, config)
        print("\nWinner Fitness:{}".format(win_fit))

    def run(self, config_file, num_generations, prev_checkpoint=None):
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        if prev_checkpoint is None:
            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(config)
        else:
            # Create the population, based upon a previous checkpoint
            p = neat.Checkpointer.restore_checkpoint(prev_checkpoint)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        if self.checkpoint_prefix is None:
            p.add_reporter(neat.Checkpointer(self.checkpoint_freq))
        else:
            p.add_reporter(neat.Checkpointer(self.checkpoint_freq,
                           filename_prefix=self.checkpoint_prefix))

        # Run for up to the given number of generations.
        if self.threads is None or self.threads == 1:
            winner = p.run(self.eval_many_genome, num_generations)
        else:
            print("USING THREADED EVALUATOR")
            pe = neat.ThreadedEvaluator(self.threads, self.eval_genome)
            winner = p.run(pe.evaluate, num_generations)

        # node_names = {-1:'PY', -2:'BY', -3:'BX', 0:'NOOP', 1:'LEFT', 2:'RIGHT'}
        # visualize.draw_net(config, winner, True, node_names=node_names)
        # visualize.plot_stats(stats, ylog=False, view=True)
        # visualize.plot_species(stats, view=True)

        # Display the winning genome.
        # print('\nBest genome:\n{!s}'.format(winner))
        input("Training Finished Press Enter to see performance...")
        # Show output of the most fit genome against training data.
        print('\nOutput:')
        # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        win_fit = self.disp_genome(winner, config)
        print("\nWinner Fitness:{}".format(win_fit))
