import random
import numpy as np
import matplotlib.pyplot as plt

# Parametry
N = 10  # liczba wierszy
M = 10  # liczba kolumn
START = (0, 0)  # Punkt startowy
GOAL = (9, 9)  # Punkt docelowy
POPULATION_SIZE = 50  # Rozmiar populacji
GENE_LENGTH = 20  # Długość sekwencji komend (genomu)
MUTATION_RATE = 0.1  # Szansa mutacji
GENERATIONS = 100  # Liczba generacji

# Mapa - 0 to wolne miejsce, 1 to przeszkoda
mapa = np.zeros((N, M))
mapa[3, 3:7] = 1
mapa[6, 2:8] = 1

# Możliwe ruchy
MOVES = ["straight", "right", "left", "u-turn"]


# Funkcje pomocnicze
def random_genome():
    """Generuje losowy genom (trasę)."""
    return [random.choice(MOVES) for _ in range(GENE_LENGTH)]


def move_robot(position, direction):
    """Oblicza nową pozycję po wykonaniu ruchu."""
    x, y = position
    if direction == "straight":
        x += 1
    elif direction == "right":
        y += 1
    elif direction == "left":
        y -= 1
    elif direction == "u-turn":
        x -= 1
    return max(0, min(N - 1, x)), max(0, min(M - 1, y))  # Ograniczenie do granic mapy


def evaluate_fitness(genome):
    """Ocenia jakość (fitness) genomu jako odległość od celu."""
    position = START
    for command in genome:
        position = move_robot(position, command)
        if mapa[position] == 1:  # Zderzenie z przeszkodą
            return float('inf')
    return np.linalg.norm(np.array(position) - np.array(GOAL))  # Odległość od celu


def select_parents(population):
    """Selekcja losowa osobników z preferencją lepszych rozwiązań."""
    population.sort(key=lambda x: x[1])
    return population[:2]  # Wybiera dwa najlepsze osobniki


def crossover(parent1, parent2):
    """Krzyżowanie dwóch rodziców."""
    idx = random.randint(1, GENE_LENGTH - 1)
    child1 = parent1[:idx] + parent2[idx:]
    child2 = parent2[:idx] + parent1[idx:]
    return child1, child2


def mutate(genome):
    """Mutacja genomu."""
    for i in range(GENE_LENGTH):
        if random.random() < MUTATION_RATE:
            genome[i] = random.choice(MOVES)
    return genome


# Inicjalizacja populacji
population = [(random_genome(), float('inf')) for _ in range(POPULATION_SIZE)]

# Algorytm genetyczny
for generation in range(GENERATIONS):
    # Ocena populacji
    population = [(genome, evaluate_fitness(genome)) for genome, _ in population]

    # Sprawdzanie najlepszego rozwiązania
    best_genome, best_fitness = min(population, key=lambda x: x[1])
    if best_fitness == 0:  # Dotarcie do celu
        print(f"Osiągnięto cel w generacji {generation}")
        break

    # Tworzenie nowej populacji
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        # Selekcja i krzyżowanie
        parent1, parent2 = select_parents(population)
        child1, child2 = crossover(parent1[0], parent2[0])

        # Mutacja
        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append((child1, float('inf')))
        new_population.append((child2, float('inf')))

    population = new_population


# Wizualizacja znalezionej trasy
def plot_map_with_path(genome):
    plt.figure(figsize=(6, 6))
    plt.imshow(mapa, cmap='gray_r')
    plt.plot(START[1], START[0], 'go', label="Start")  # Start
    plt.plot(GOAL[1], GOAL[0], 'ro', label="Cel")  # Goal

    # Śledzenie pozycji robota
    position = START
    path_x, path_y = [position[1]], [position[0]]

    for command in genome:
        position = move_robot(position, command)
        path_x.append(position[1])
        path_y.append(position[0])
        if mapa[position] == 1:  # Zderzenie z przeszkodą
            break

    plt.plot(path_x, path_y, 'b-', label="Trasa robota")  # Rysowanie trasy
    plt.legend()
    plt.title("Mapa z przeszkodami i trasą robota")
    plt.xlabel("Kolumny")
    plt.ylabel("Wiersze")
    plt.grid()
    plt.show()


# Wyświetlenie mapy z trasą
print("Najlepsza trasa:", best_genome)
print("Odległość od celu:", best_fitness)
plot_map_with_path(best_genome)
