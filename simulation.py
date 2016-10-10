#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random
from matplotlib.widgets import Button


M = 100
N = 100

FOOD = int(0.33 * (M * N))

INITIAL_POPULATION_DENSITY = 0.01
INITIAL_SHARE_RATIO = 0.0
MUTATION_RATE = 0.001

EMPTY = 0
HOG = 1
SHARER = 2


def neighbors(population, i, j):
  """a list of the 4 neighbors around coordinates (i, j) in the population matrix"""
  return [
    population[i, (j + 1) % N],
    population[i, (j - 1) % N],
    population[(i + 1) % M, j],
    population[(i - 1) % M, j]
  ];

def populated(population, i, j):
  return 1 if population[i % M, j % N] != EMPTY else 0

def redistributeFood(population, food):
  """ sharers send food to their neighbors

  Maybe this is more of a "behavior strategy" step, ie this type of sharing is just one of many alternatives"
  """
  newFood = np.zeros(shape=(M, N));
  for i in range(M):
    for j in range(N):
      cell = population[i, j]
      if cell == EMPTY:
        newFood[i, j] += food[i, j]
      elif cell == HOG:
        newFood[i, j] += food[i, j]
      elif cell == SHARER:
        neighborCount = len(filter(isPopulated, neighbors(population, i, j))) 
        
        if food[i, j] > 1 and neighborCount > 0:
          newFood[i, j] = 1
          ration = (food[i, j] * 1.0 - 1.0) / neighborCount

          # feed my neighbors
          if populated(population, i, j + 1):
            newFood[i, (j + 1) % N] += ration
          if populated(population, i, j - 1):
            newFood[i, (j - 1) % N] += ration
          if populated(population, i + 1, j):
            newFood[(i + 1) % M, j] += ration
          if populated(population, i - 1, j):
            newFood[(i - 1) % M, j] += ration
        else:
          newFood[i, j] = food[i, j]
  return newFood

def distributeFood(shape=(M, N), food=FOOD):
  """ randomly distribute food into an mxn matrix """
  foodDistribution = np.zeros(shape=shape);
  for _ in range(food):
    i = random.randint(0, M - 1)
    j = random.randint(0, M - 1)
    foodDistribution[i, j] += 1
  return foodDistribution

def distributePopulation(density = INITIAL_POPULATION_DENSITY, proportion = INITIAL_SHARE_RATIO):
  """ create an initial population distribution in a MxN matrix. """
  population = np.zeros(shape=(M, N))
  for i in range(M):
    for j in range(N):
      if random.uniform(0, 1) < density:
        population[i, j] = SHARER if random.uniform(0, 1) < proportion else HOG
  return population

def cullPopulation(population, food):
  """ kill individuals who don't have enough food """
  newPopulation = np.zeros(shape=(M, N))
  for i in range(M):
    for j in range(N):
      newPopulation[i, j] = population[i, j] if random.uniform(0, 1) < food[i, j] else EMPTY
  return newPopulation

def isSharer(x):
  return x == SHARER

def isHog(x):
  return x == HOG

def isEmpty(x):
  return x == EMPTY

def isPopulated(x):
  return not isEmpty(x)

def propagatePopulation(population):
  """ given a population distribution calculate the new population.
  
  if a cell is empty, it might become populated based on the presence of neighborss.
  """
  newPopulation = np.zeros(shape=(M, N))
  for i in range(M):
    for j in range(N):
      if population[i, j] == EMPTY:
        #print i, j, 'is empty'
        ns = neighbors(population, i, j)

        shareCount = len(filter(isSharer, ns))

        hogCount   = len(filter(isHog, ns))
        
        totalCount = shareCount + hogCount;

        if totalCount == 0:
          newPopulation[i, j] = EMPTY
        else:
          shareRatio = (shareCount * 1.0 / totalCount)
          
          shareRatioWithMutation = shareRatio + random.uniform(-MUTATION_RATE, MUTATION_RATE)

          #print 'total =', totalCount, ', share =', shareCount, ', hog =', hogCount, ', mutation =', shareRatioWithMutation
          newPopulation[i, j] = SHARER if random.uniform(0, 1) < shareRatioWithMutation else HOG

      else:
        newPopulation[i, j] = population[i, j]
  return newPopulation

def populationStats(population):
  l = population.flatten()
  sharers = len(filter(isSharer, l))
  hogs = len(filter(isHog, l))
  return sharers, hogs

def step(population):
  # food is randomlly distributed in each step.
  food = distributeFood()
  # sharers can redistribute the food (behaviorial strategy)
  redistributedFood = redistributeFood(population, food)
  # TODO: food redistribution seems to result in a net loss of food... Where did it go? Must be a bug. Rounding errors? Seems too much
  print 'food:', food.sum(), '; redistributed:', redistributedFood.sum()
  # the population is culled, those without food die. (AKA selection, differential success)
  culledPopulation = cullPopulation(population, redistributedFood)
  # the next generation is determined based on those that survived the culling (AKA reproduction)
  nextGeneration = propagatePopulation(culledPopulation)
  return nextGeneration

def testFoodPlot():
  matrix = distributeFood()
  plot(matrix)

def plot(matrix):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest')
  #plt.colorbar()
  plt.legend(loc='best')
  plt.ion()
  plt.show()

global_population = distributePopulation()
generation = 0
xs = []
hogCounts = []
sharerCounts = []

def animate():  
  global global_population
  global generation
  global_population = distributePopulation()
  generation = 0
  def update(_):
    global global_population
    global generation
    global_population = step(global_population)

    sharerCount, hogCount = populationStats(global_population)
    print 'generation:', generation, ', hogs:', hogCount, ', sharers:', sharerCount, ', population:', sharerCount + hogCount
    sharerCounts.append(sharerCount)
    hogCounts.append(hogCount)

    generation += 1
    xs.append(generation)

    #hogLine.set_data(xs, hogCounts)

    #shareLine.set_data(xs, sharerCounts)

    im.set_array(global_population)
    return im

  fig = plt.figure()
  #fig.subplots_adjust(bottom=0.2)
  ax = fig.add_subplot(1, 1, 1)


  ax.set_aspect('equal')
  ax.set_xbound(0, M)
  ax.set_ybound(0, N)

  #ax.set_title('Foo Bar')
  cmap = plt.cm.get_cmap('viridis', 3) 
  im = plt.imshow(global_population, interpolation='nearest', vmin=0, vmax=2, cmap=cmap)

  sharePatch = mpatches.Patch(color=cmap(SHARER), label='Share')
  hogPatch = mpatches.Patch(color=cmap(HOG), label='Hog')
  plt.legend(handles=[sharePatch, hogPatch], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

 

  #ax2 = fig.add_subplot(2, 2, 2)
  #hogLine, = plt.plot(xs, hogCounts)
  #shareLine, = plt.plot(xs, sharerCounts)
  #plt.xlabel('Generation')
  #plt.ylabel('Population')
  
  plt.ion()
  ani = animation.FuncAnimation(fig, update, interval=250, blit=False)

  #def clicked(event):
  #  print 'button clicked'

  #axReset = plt.axes([0.7, 0.05, 0.1, 0.075])
  #buttonReset = Button(axReset, 'Reset')
  #buttonReset.on_clicked(clicked) 

  #Writer = animation.writers['ffmpeg']
  #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  #ani.save('ess.mp4', writer=writer)

  plt.show()
  return ani

a = animate()
