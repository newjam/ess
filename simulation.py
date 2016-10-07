#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random


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
  return [
    population[i, (j + 1) % N],
    population[i, (j - 1) % N],
    population[(i + 1) % M, j],
    population[(i - 1) % M, j]
  ];


def populated(population, i, j):
  return 1 if population[i % M, j % N] != EMPTY else 0

def redistributeFood(population, food):
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

def distributeFood():
  foodDistribution = np.zeros(shape=(M, N));
  for _ in range(FOOD):
    i = random.randint(0, M - 1)
    j = random.randint(0, M - 1)
    foodDistribution[i, j] += 1
  return foodDistribution

def distributePopulation(density = INITIAL_POPULATION_DENSITY, proportion = INITIAL_SHARE_RATIO):
  population = np.zeros(shape=(M, N))
  for i in range(M):
    for j in range(N):
      if random.uniform(0, 1) < density:
        population[i, j] = SHARER if random.uniform(0, 1) < proportion else HOG
  return population

def cullPopulation(population, food):
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

def step(population):
  food = distributeFood()
  
  redistributedFood = redistributeFood(population, food)
  print 'food:', food.sum(), '; redistributed:', redistributedFood.sum()
  culledPopulation = cullPopulation(population, redistributedFood)
  nextGeneration = propagatePopulation(culledPopulation)

  l = nextGeneration.flatten()
  sharers = len(filter(isSharer, l))
  hogs = len(filter(isHog, l))

  print 'hogs:', hogs, ', sharers:', sharers

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

def animate():  
  global global_population
  global_population = distributePopulation()
  def update(_):
    global global_population
    global_population = step(global_population)
    im.set_array(global_population)
    return im,
  
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_aspect('equal')
  ax.set_xbound(0, M)
  ax.set_ybound(0, N)
  cmap = plt.cm.get_cmap('viridis', 3) 
  im = plt.imshow(global_population, interpolation='nearest', vmin=0, vmax=2, cmap=cmap)

  sharePatch = mpatches.Patch(color=cmap(SHARER), label='Share')
  hogPatch = mpatches.Patch(color=cmap(HOG), label='Hog')
  plt.legend(handles=[sharePatch, hogPatch], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

  #plt.colorbar()
  plt.ion()
  ani = animation.FuncAnimation(fig, update, interval=250, blit=False)

  #Writer = animation.writers['ffmpeg']
  #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
  #ani.save('ess.mp4', writer=writer)

  plt.show()
  return ani

a = animate()
