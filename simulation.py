#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

M = 100
N = 100

FOOD = int(0.33 * (M * N))

INITIAL_POPULATION_DENSITY = 0.01
INITIAL_SHARE_RATIO = 0.5
MUTATION_RATE = 0.1

EMPTY = 0
HOG = 1
SHARER = 2

def populated(population, i, j):
  return 1 if population[i % M, j % N] > 0 else 0

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
        neighborCount = populated(population, i, j + 1) \
                      + populated(population, i, j - 1) \
                      + populated(population, i + 1, j) \
                      + populated(population, i - 1, j)
        
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

def propagatePopulation(population):
  newPopulation = np.zeros(shape=(M, N))
  for i in range(M):
    for j in range(N):
      if population[i, j] == EMPTY:
        shareCount = 1 if population[i, (j + 1) % N] == SHARER else 0 \
                   + 1 if population[i, (j - 1) % N] == SHARER else 0 \
                   + 1 if population[(i + 1) % M, j] == SHARER else 0 \
                   + 1 if population[(i - 1) % M, j] == SHARER else 0

        hogCount   = 1 if population[i, (j + 1) % N] == HOG else 0 \
                   + 1 if population[i, (j - 1) % N] == HOG else 0 \
                   + 1 if population[(i + 1) % M, j] == HOG else 0 \
                   + 1 if population[(i - 1) % M, j] == HOG else 0
        
        totalCount = shareCount + hogCount

        if totalCount == 0:
          newPopulation[i, j] = EMPTY
        else:
          shareRatio = (shareCount * 1.0 / totalCount)
          shareRatioWithMutation = shareRatio + random.uniform(-MUTATION_RATE, MUTATION_RATE)

          newPopulation[i, j] = SHARER if random.uniform(0, 1) < shareRatioWithMutation else HOG

      else:
        newPopulation[i, j] = population[i, j]
  return newPopulation

def step(population):
  food = distributeFood()
  
  redistributedFood = redistributeFood(population, food)
  print 'food:', food.sum(), '; redistributed:', redistributedFood.sum()
  culledPopulation = cullPopulation(population, redistributedFood)
  return propagatePopulation(culledPopulation)

def testFoodPlot():
  matrix = distributeFood()
  plot(matrix)

def plot(matrix):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest')
  plt.colorbar()
  plt.ion()
  plt.show()


population = distributePopulation()

def animate():  
  global population
  population = distributePopulation()
  def update(_):
    global population
    population = step(population)
    im.set_array(population)
    return im,
  
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_aspect('equal')
  im = plt.imshow(population, interpolation='nearest')
  plt.colorbar()
  plt.ion()
  ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
  plt.show()
  return ani


