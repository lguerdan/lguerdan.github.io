---
layout: page
title: The Fittest Mario
description: An evolutionary computation approach to Super Mario Bros
img: /assets/img/smb.png
importance: 1
---

This project was a collaboration with **Pooneh Safayenikoo**, **Pedro Carrion Castagnola**, **Marshall Lindsay**, **Jeffrey Ruffolo** at the University of Missouri. My role involved developing a generic state representation and optimization framework for the game that could be extended via evolutionary computation techniques. My teammates ran experiments with with specific aspects of the solver, including the fitness function, parent selection, mutations, and crossover methods.  

Here is a demo video I made about the system (animations courtesy of Jeffrey Ruffolo), tailored to high school science students. 

<figure>
    <iframe src="https://drive.google.com/file/d/1q3ZSv5JlkcVEg80W5McjOVl_FKM08goK/preview" width="110%" height="450px"></iframe>
</figure>


Super Mario Bros is a classic platformer video game in which the player tries to navigate Mario to the flag at the end of the stage. During this time, the player must successfully overcome obstacles and terrains in order to progress. Our project implements an evolutionary computation approach to finding a sequence of player actions to beat the game. We experiment with several operators for parent selection, crossover, mutation, and fitness evaluation. Results show successful convergence towards winning action sequences, with convergence rate responding to changes in strategies. We also observed different gameplay behaviors for chromosomes evaluated using different fitness functions.

# Chromosome representation
The chromosome is represented as a sequence of game actions taken by the player (such as up, left, or right), which when emulated result in a fitness score for the given chromosome. We also used the index of game over for crossover and mutation methods because after this index the actions do not influence on the fitness. The chromosome in our implementation is an object that contains: a list of action sequences (integers), fitness score and index of game over. Figure 1 shows a representation of the chromosome.

<div class="row">
    <div class="col-sm mt-3 mt-md-0 margin-top center">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/ec1.png' | relative_url }}" alt=""/>
    </div>
</div>
<div class="caption">
    Figure 1.
</div>

# Simulation environment
We used gym-super-mario-bros, an [OpenAI Gym environment](https://gym.openai.com/) for Super Mario Bros on The Nintendo Entertainment System (NES) using the [nes-py emulator](https://github.com/Kautenja/gym-super-mario-bros). This environment provides an easy-to-use, high-level API for making commands in Super Mario Bros. Figure 2 shows the general pipeline of our experiments. 

<div class="row">
    <div class="col-sm mt-2 mt-md-0 margin-top center">
        <img class="img-fluid rounded z-depth-1" src="{{ '/assets/img/ec2.png' | relative_url }}" alt=""/>
    </div>
</div>
<div class="caption">
    Figure 2.
</div>


To implement this experiment pipeline, I made a simple boilerplate Python wrapper, `MSBGeneticOptimizerEnv`, that maintains the chomosome state and runs chromosome selection, crossover, and mutation:


{% highlight python linenos %}
from __future__ import print_function

import multiprocessing, time, functools, gym_super_mario_bros, os, csv

from multiprocessing import Pool
from contextlib import contextmanager
from itertools import product
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

try:
   import cPickle as pickle
except:
   import pickle


#MSBGeneticOptimizerEnv contains boilerplate code for the project
class MSBGeneticOptimizerEnv(object):
   """An environment wrapper for genetically optimizing mario smash brothers simulation ."""

   def __init__(self, max_steps=3000, num_chromosomes=40, action_encoding=RIGHT_ONLY, render=False, fitness_strategy="x_pos", session_file="", world=1, stage=1, version=0, noFrameSkip=False):
      if session_file != "":
         self.load_optimizer(session_file)
      else:
         self.max_steps = max_steps
         self.action_encoding = action_encoding
         self.render = render
         self.fitness_strategy = fitness_strategy  #score, x_pos, time, coins
         self.num_chromosomes = num_chromosomes
         self.world = world
         self.stage = stage
         self.version = version
         self.noFrameSkip = 'NoFrameskip' if noFrameSkip else ''
         self.init_chromosomes()

   def init_chromosomes(self):
      """Creates a new set of genes based on the number of parents fed in"""
      self.chromosomes = []
      for i in range(self.num_chromosomes):
         chromosome = [np.random.randint(0,len(self.action_encoding), self.max_steps), -1, -1]
         self.chromosomes.append(chromosome)
      self.evaluate_chromosomes()

   def save_optimizer(self, fname):
      print("saving optimizer state to ",fname)
      optimizer_state = Optimizer(self.max_steps, self.num_chromosomes, self.action_encoding, self.render, self.fitness_strategy, self.chromosomes, self.world, self.stage, self.version, self.noFrameSkip)
      with open(fname, "wb") as f:
         pickle.dump(optimizer_state, f)

   def load_optimizer(self, fname):
      print("loading optimier state from ",fname)
      with open(fname, "rb") as f:
         optimizer = pickle.load(f)
         self.max_steps = optimizer.max_steps
         self.action_encoding = optimizer.action_encoding
         self.render = optimizer.render
         self.fitness_strategy = optimizer.fitness_strategy
         self.num_chromosomes = optimizer.num_chromosomes
         self.chromosomes = optimizer.chromosomes
         self.world = optimizer.world
         self.stage = optimizer.stage
         self.version = optimizer.version
         self.noFrameSkip = optimizer.noFrameSkip


   def run_generations(self, ngens, fname):

      headers = ['generation', 'chromosome_num', self.fitness_strategy, 'avg_fitness']

      #If logging for the first time, set up csv file
      if fname:
         print("logging progress to " + fname)

         with open (fname, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
            writer.writeheader()

      for gen in range(ngens):
         self.new_generation()
         self.evaluate_chromosomes()
         max_fitness, max_fitness_ix = self.get_max_fitness_chromosome()
         avg_fitness = self.get_avg_chromosome_fitness()

         #If writing progress to output, add this generation
         if fname:
            with open (fname, 'a') as csvfile:
               writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
               writer.writerow({'generation': gen, 'chromosome_num': max_fitness_ix, self.fitness_strategy: max_fitness, 'avg_fitness': avg_fitness})

         print("\n#################################")
         print("GENERATION",gen,"COMPLETE")
         print("Highest chromosome: ",max_fitness_ix,", fitness:",max_fitness)
         print("Average fitness: ", avg_fitness)
         print("####################################\n\n\n")


   def get_max_fitness_chromosome(self):
      """returns highest fitness of current chromosomes, along with its index"""
      max_fitness = -1
      max_fitness_ix = -1
      max_chromosome = []
      for cix, chromosome in enumerate(self.chromosomes):
         if chromosome[1] > max_fitness:
            max_fitness = chromosome[1]
            max_fitness_ix = cix

      return max_fitness, max_fitness_ix

   def get_avg_chromosome_fitness(self):
      total_fitness = 0
      for chromosome in self.chromosomes:
         total_fitness += chromosome[1]

      return int(total_fitness / len(self.chromosomes))

   @abstractmethod
   def new_generation(self):
      """
      Based on a chromosomes structure, updates the chromosomes by natural selection rules
      This is where the bulk of the evolutionary computation code will go
      Update here
      """
      #
      pass
      # For now now selection occurs, just keep current chromosomes


   def run_top_chromosome(self, render=False):
      """
      Retrieve the best-performing chromosome and play it.
      Override render argument in case only want to visualize on test round
      """
      max_fitness, max_fitness_ix = self.get_max_fitness_chromosome()

      if max_fitness == -1:
         print('Run top chromosome error: no fitnesses have been computed')

      with mariocontext(self) as env:
         done = True
         for step, action in enumerate(self.chromosomes[max_fitness_ix][0]):

            state, reward, done, info = env.step(action)
            if done or info['flag_get']:
               return

            if render: env.render()


   def evaluate_chromosome(self, input_tuple):
      """Evaluates a chromosome for it's fitness value and index of death"""

      chromosome_num, chromosome = input_tuple

      if chromosome[1] != -1:
         return chromosome

      with mariocontext(self) as env:

         best_fitness_step = 0

         state = env.reset()
         #Main evaluation loop for this chromosome
         for step, action in enumerate(chromosome[0]):

            #take step
            state, reward, done, info = env.step(action)

            if (info[self.fitness_strategy] > best_fitness_step):
               best_fitness_step = step

            #died or level beat
            if (done or info['flag_get']):
               break

            #print progress
            if step % 50 == 0:
               print("chromosome:",chromosome_num," step:", step," action:",action, "info:",info)

            #display on screen
            if self.render:
               env.render()

         chromosome[1], chromosome[2] = info[self.fitness_strategy], best_fitness_step

         print("chromosome",chromosome_num," done fitness ",self.fitness_strategy ,"= ",info[self.fitness_strategy])
         return chromosome


   def evaluate_chromosomes(self):
      """
      Given a gene structure, evaluates all genes for their fitness and stores it in the stucture
      This is what actually runs the training simulation
      Input: a gene structure with (possibly) empty fitnesses
      Output: a gene structure with computed fitnesses and index of death
      """

      bound_instance_method_alias = functools.partial(_instance_method_alias, self)
      with poolcontext(1) as pool:
         self.chromosomes = pool.map(bound_instance_method_alias, enumerate(self.chromosomes))


class Optimizer():
   """A basic container class for saving optimizer contents"""

   def __init__(self, max_steps, num_chromosomes, action_encoding, render, fitness_strategy, chromosomes, world, stage, version, noFrameSkip):
      self.max_steps = max_steps
      self.num_chromosomes = num_chromosomes
      self.action_encoding = action_encoding
      self.render = render
      self.fitness_strategy = fitness_strategy
      self.chromosomes = chromosomes
      self.world = world
      self.stage = stage
      self.version = version
      self.noFrameSkip = noFrameSkip


@contextmanager
def poolcontext(*args, **kwargs):
   pool = multiprocessing.Pool(*args, **kwargs)
   yield pool
   pool.terminate()

@contextmanager
def mariocontext(marioEnv):
   mario_env = 'SuperMarioBros' + marioEnv.noFrameSkip + '-' + str(marioEnv.world) + '-' + str(marioEnv.stage) + '-v' + str(marioEnv.version)
   env = gym_super_mario_bros.make(mario_env)
   env = BinarySpaceToDiscreteSpaceEnv(env, marioEnv.action_encoding)
   yield env
   env.close()


def _instance_method_alias(obj, arg):
   """
   Alias for instance method that allows the method to be called in a
   multiprocessing pool
   """
   return obj.evaluate_chromosome(arg)

{% endhighlight %}


This environment can be easily extended to customize crossover, parent selection, and mutation methods. Here is an example of an `EvolutionEnv` extending the `MSBGeneticOptimizerEnv`:


{% highlight python linenos %}
from msb_genetic_optimizer_env import MSBGeneticOptimizerEnv
import numpy as np


##The below class inherits everything from MSBGeneticOptimizerEnv, all you will need to do is modi
class EvolutionEnv(MSBGeneticOptimizerEnv):
   # Arguments: max_steps=10000, num_chromosomes=4, action_encoding=SIMPLE_MOVEMENT, render=False, reward="score", session_file=""
   def __init__(self, *args, **kwargs):
      super(EvolutionEnv, self).__init__(*args, **kwargs)

   def new_generation(self):
      """
      Based on a chromosomes structure, updates the chromosomes by natural selection rules
      This is where the bulk of the evolutionary computation code will go
      We will need to modify the chromosome structure in some
      """
        
      # elite selection for parent
      parents = self.select_parents(3, int(self.num_chromosomes/2))
      offspring = []

      # Crossover
      for i in range(int(np.floor(len(parents) / 2))):
         offspring.extend(self.crossover_chromosome_pair(parents[2 * i], parents[2 * i + 1],
                              point_num=2, points_before_death=True, normal_dist=True))

      # Mutation
      # using lambda = mu (numberof parents equal to the number of offsprings)
      # For each selected parent, will mutate around 20% of the max_steps actions
      mutations = round(0.2 * self.max_steps)
      for chromosome in parents:
         child = [chromosome[0].copy(), -1, chromosome[2]]
         self.mutate_chromosome(child, mutations)
         offspring.append(child)

      # (mu, lambda) Using only offspring to populate new generation here
      # self.chromosomes = offspring

      # (mu + lambda) Using parents + offspring to populate new generation here
      self.chromosomes = parents + offspring

   ###
   #  Basic implementation of parent selection
   #  - selection_type: 0: shuffle, 1: elite
   #  - mu: number of parents to select
   ###
   def select_parents(self, selection_type=1, mu=1):
      def shuffle_selection():
         np.random.shuffle(self.chromosomes)
         return self.chromosomes[:mu]

      def elite_selection(mu):
         parents = []
         # select mu best chromosomes
         best_chromosomes_index = sorted(range(len(self.chromosomes)), key=lambda x: self.chromosomes[x][1],
                                         reverse=True)
         # delete from chromosome list if it is not within the mu best
         for i in range(mu):
            parents.append(self.chromosomes[best_chromosomes_index[i]])
         return parents

      def linear_ranking_selection(mu):
         parents = []
         best_chromosomes_index = sorted(range(len(self.chromosomes)), key=lambda x: self.chromosomes[x][1], reverse=True)
         s = []
         s.append(0)
         for i in range(self.num_chromosomes):
            s.append(s[i]+((1.0/self.num_chromosomes)*(1.5-((i)/(self.num_chromosomes-1.0)))))
         for i in range(mu):
            r=s[self.num_chromosomes]*np.random.random_sample()
            for i in range(self.num_chromosomes):
               if s[i] <= r < s[i+1]:
                  parents.append(self.chromosomes[best_chromosomes_index[self.num_chromosomes-(i+1)]])
         return parents

      def roulette_wheel_selection(mu):
         parents = []
         total=0
         for i in range(self.num_chromosomes):
            total=total+self.chromosomes[i][1];
         s = []
         s.append(0)
         for i in range(self.num_chromosomes):
            s.append(s[i]+((self.chromosomes[i][1])/(float)(total)))
         for i in range(mu):
            r=np.random.random_sample()
            for i in range(self.num_chromosomes):
               if s[i] <= r < s[i+1]:
                  parents.append(self.chromosomes[i])
         return parents

      if selection_type == 0:
         return shuffle_selection()
      elif selection_type == 1:
         return elite_selection(mu)
      elif selection_type == 2:
         return linear_ranking_selection(mu)
      elif selection_type == 3:
         return roulette_wheel_selection(mu)

   def crossover_chromosome_pair(self, parent1, parent2, point_num=1, points_before_death=False, normal_dist=False):
      ###
      #	Implementation of crossover functionality
      #	- points: number of points to use in crossover
      #	- points_before_death: only consider points upto sooner death
      #	- normal_dist: sample crossover points from normal distribution
      ###
      def positive_normal():
         x = abs(np.random.randn() / 3)
         return max(1 - x, 0)

      point_range_max = min(len(parent1[0]), len(parent2[0])) if not points_before_death \
         else max(parent1[2], parent2[2])
      random = np.random.rand if not normal_dist else positive_normal

      crossover_points = [point_range_max]
      for i in range(point_num):
         crossover_points.append(int(round(point_range_max * random())))
      crossover_points.sort(reverse=True)

      child1, child2 = [parent1[0].copy(), -1, -1], [parent2[0].copy(), -1, -1]
      while len(crossover_points) > 1:
         point1 = crossover_points.pop()
         point2 = crossover_points.pop()
         child1[0][point1:point2] = parent2[0][point1:point2].copy()
         child2[0][point1:point2] = parent1[0][point1:point2].copy()

      return [child1, child2]

   def mutate_chromosome(self, chromosome, mutations=1):
   ###
   # Implementation of mutation operator
   # Mutations: number of mutations to make
   ###
      for i in range(int(mutations)):
         #mutation index: triangular distribution with: bound left=0, bound right and mode=index of game over
         mutation_index = int(np.ceil(np.random.triangular(left=0, mode=1, right=1) * chromosome[2]))
         #new_action: random within the allowed possible actions
         actions_size = len(self.action_encoding)
         #new_action = np.random.randint(low=0, high=actions_size)
         #new: prioritize actions between 1 to 4 (going right)
         new_action = np.random.randint(low=0, high=(actions_size+4))
         if (new_action >= actions_size):
            new_action = new_action - actions_size + 1
         chromosome[0][mutation_index] = new_action
      chromosome[1], chromosome[2] = -1, -1

{% endhighlight %}

After this high-level wrapper is complete, it can easily be used in a short script. Here is an example experiment: 

{% highlight python linenos %}
#create a new optimizer environment and initialize data structure
optimizer = SelectionEnv(max_steps = 500, num_chromosomes=6, render=True)

#run the optimizer for the desired number of generations
optimizer.run_generations(3)

#serialize the data structure to a file. Pickle for effeciency 
optimizer.save_optimizer('mario-4-chromosome.p')

#see how the top performing chromosome looks in the simulator 
optimizer.run_top_chromosome(render=True)

#To load a previous environment, either create a new optimizer with the filename,or call the load optimizer function directly. Note this will overwrite the current state
optimizer2 = SelectionEnv(session_file ="mario-4-chromosome.p")
load the previous session and keep training
optimizer2.run_generations(1)
optimizer2.save_optimizer('mario-4-chromosome2.p') #save to an updated file 
{% endhighlight %}


My teammates then extended this environment to examine different evolutionary computation simulations. See our [poster here]({{site.baseurl}}/assets/pdf/mario_ec_poster.pdf){:target="_blank"} for more information about these aspects of the project!

