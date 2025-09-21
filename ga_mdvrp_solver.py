# Genetic Algorithm for Multi-Depot Vehicle Routing Problem (MDVRP)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import sqrt

# -------------------------------
# 1. Problem Statement
# -------------------------------
"""
We solve a simplified version of the Multi-Depot Vehicle Routing Problem (MDVRP),
where each customer must be assigned to a depot and a route must be planned for each depot's vehicle
in a way that minimizes the total travel distance of all routes.
"""

# Load a sample MDVRP problem from the dataset
file_path = '/mnt/data/19MDVRP Problem Sets.xlsx'
df = pd.read_excel(file_path, sheet_name='Problem 9')

# Separate customers and depots
customers = df[['Customer Number', 'x coordinate', 'y coordinate']].dropna().astype({'Customer Number': int})
depots = df[['Depot x coordinate', 'Depot y coordinate']].dropna().drop_duplicates().reset_index(drop=True)

# -------------------------------
# 2. Chromosome Representation
# -------------------------------
"""
Each chromosome is a permutation of customer indices, split into subroutes (one per depot).
"""

def distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def assign_customers_to_depots():
    assignments = {}
    for idx, cust in customers.iterrows():
        c_coord = (cust['x coordinate'], cust['y coordinate'])
        min_dist = float('inf')
        nearest_depot = None
        for depot_id, depot in depots.iterrows():
            d_coord = (depot['Depot x coordinate'], depot['Depot y coordinate'])
            dist = distance(c_coord, d_coord)
            if dist < min_dist:
                min_dist = dist
                nearest_depot = depot_id
        assignments.setdefault(nearest_depot, []).append(int(cust['Customer Number']))
    return assignments

# -------------------------------
# 3. Genetic Operators
# -------------------------------

def selection(population, fitnesses):
    selected = random.choices(population, weights=[1/f for f in fitnesses], k=2)
    return selected

def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1)-2)
    child1 = parent1[:cut] + [g for g in parent2 if g not in parent1[:cut]]
    child2 = parent2[:cut] + [g for g in parent1 if g not in parent2[:cut]]
    return child1, child2

def mutate(chromosome, mutation_rate=0.05):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome)-1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# -------------------------------
# 4. Fitness Function
# -------------------------------

def evaluate_fitness(chromosome, assignments):
    total_distance = 0
    pointer = 0
    for depot_id, assigned_customers in assignments.items():
        depot_coord = (depots.iloc[depot_id]['Depot x coordinate'], depots.iloc[depot_id]['Depot y coordinate'])
        subroute = [c for c in chromosome if c in assigned_customers]
        if not subroute:
            continue
        # Route: depot -> c1 -> c2 -> ... -> depot
        route_coords = [depot_coord]
        for cid in subroute:
            c = customers[customers['Customer Number'] == cid].iloc[0]
            route_coords.append((c['x coordinate'], c['y coordinate']))
        route_coords.append(depot_coord)
        total_distance += sum(distance(route_coords[i], route_coords[i+1]) for i in range(len(route_coords)-1))
    return total_distance

# -------------------------------
# 5. Evolutionary Loop
# -------------------------------

def genetic_algorithm(assignments, population_size=50, generations=100, mutation_rate=0.1):
    all_customers = sum(assignments.values(), [])
    population = [random.sample(all_customers, len(all_customers)) for _ in range(population_size)]
    best_fitness_list = []
    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitnesses = [evaluate_fitness(ind, assignments) for ind in population]
        new_population = []
        for _ in range(population_size // 2):
            p1, p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, mutation_rate))
            new_population.append(mutate(c2, mutation_rate))
        population = new_population

        gen_best = min(zip(population, fitnesses), key=lambda x: x[1])
        if gen_best[1] < best_fitness:
            best_solution = gen_best[0]
            best_fitness = gen_best[1]
        best_fitness_list.append(best_fitness)

    return best_solution, best_fitness, best_fitness_list

# -------------------------------
# 6. Run and Visualize
# -------------------------------

assignments = assign_customers_to_depots()
best_sol, best_fit, fitness_curve = genetic_algorithm(assignments, generations=100)

print(f"Best Total Distance: {best_fit:.2f}")
plt.plot(fitness_curve)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("GA Optimization Progress")
plt.grid()
plt.show()
