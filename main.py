import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# Parametry
N = 50  # liczba wierszy
M = 50  # liczba kolumn
START = (0, 0)  # Punkt startowy
GOAL = (49, 49)  # Punkt docelowy
POPULATION_SIZE = 100  # Rozmiar populacji
MAX_GENE_LENTH = 300  # Długość sekwencji komend (genomu)
MUTATION_RATE = 0.1  # Szansa mutacji
GENERATIONS = 5000  # Liczba generacji
LEN_RATE = 0.01

# Mapa - 0 to wolne miejsce, 1 to przeszkoda
mapa = np.zeros((N, M))

# Generowanie losowych przeszkód
num_obstacles = 50  # Liczba przeszkód

def create_obstacle(num_obstacles, mapa):
    for _ in range(int(num_obstacles * 0.4)):
        x = random.randint(0, N-1)
        y = random.randint(0, M-1)
        while (x < 4 and y < 4) or (x > N-4 and y > M-4):
            x = random.randint(0, N-1)
            y = random.randint(0, M-1)
        mapa[x, y:y+2] = 1

        x = random.randint(0, N-1)
        y = random.randint(0, M-1)
        while (x < 4 and y < 4) or (x > N-4 and y > M-4):
            x = random.randint(0, N-1)
            y = random.randint(0, M-1)
        mapa[x:x+2, y] = 1
    
    for _ in range(int(num_obstacles * 0.1)):
        x = random.randint(10, N-5)
        y = random.randint(10, M-5)
        mapa[x:x+6, y:y+2] = 1

        x = random.randint(10, N-5)
        y = random.randint(10, M-5)
        mapa[x:x+2, y:y+6] = 1
    
while(True):
    create_obstacle(num_obstacles, mapa)
    plt.imshow(mapa, cmap='gray_r')
    plt.show()
    if input("Czy przeszkody są OK? (t/n): ") == 't':
        break

# Możliwe ruchy
MOVES = ["straight", "right", "left", "u-turn"]


# Funkcje pomocnicze
def random_genome():
    """Generuje losowy genom o zmiennej długości."""
    length = random.randint(int(np.sqrt(N*M)), MAX_GENE_LENTH)  # Możesz ustawić minimalną i maksymalną długość
    return [random.choice(MOVES) for _ in range(length)]

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
    """
    Ocenia jakość (fitness) genomu jako kombinację:
    1. Odległości od celu.
    2. Długości drogi (genomu).
    """
    position = START
    penalty = 0  # Kara za kolizję
    for command in genome:
        position = move_robot(position, command)
        if mapa[position] == 1:  # Zderzenie z przeszkodą
            penalty += 500  # Kara za każdą kolizję
            break  # Kończymy trasę przy kolizji

    # Odległość euklidesowa od celu
    distance_to_goal = np.linalg.norm(np.array(position) - np.array(GOAL))

    # Uwzględnienie długości genomu
    genome_length_penalty = len(genome) * LEN_RATE  # Waga kary za długość genomu

    # Całkowity fitness
    return distance_to_goal + genome_length_penalty + penalty

def select_parents(population):
    """Selekcja losowa osobników z preferencją lepszych rozwiązań."""
    population.sort(key=lambda x: x[1])
    return population[:2]  # Wybiera dwa najlepsze osobniki

def crossover(parent1, parent2):
    """Krzyżowanie dwóch rodziców o różnych długościach."""
    len1, len2 = len(parent1), len(parent2)
    idx1 = random.randint(0, len1 - 1)
    idx2 = random.randint(0, len2 - 1)
    
    child1 = parent1[:idx1] + parent2[idx2:]
    child2 = parent2[:idx2] + parent1[idx1:]
    
    return child1, child2

def mutate(genome):
    """Mutacja genomu o zmiennej długości."""
    new_genome = []
    for gene in genome:
        if random.random() < MUTATION_RATE:
            if random.random() < 0.5:  # Zmieniamy gen
                new_genome.append(random.choice(MOVES))
            # W przeciwnym wypadku pomijamy gen (czyli usuwamy)
        else:
            new_genome.append(gene)
    
    # Możliwość dodania nowego genu
    if random.random() < MUTATION_RATE:
        new_genome.append(random.choice(MOVES))
    return new_genome


# Inicjalizacja populacji
population = [(random_genome(), float('inf')) for _ in range(POPULATION_SIZE)]

progress = []
prog_gen = []
best_gen_video = []
last_best_genome = float('inf')
last_best_fitness = float('inf')

# Algorytm genetyczny
for generation in range(GENERATIONS):
    # Ocena populacji
    population = [(genome, evaluate_fitness(genome)) for genome, _ in population]

    # Sprawdzanie najlepszego rozwiązania
    best_genome, best_fitness = min(population, key=lambda x: x[1])

    if best_fitness < last_best_fitness:
        progress.append(best_fitness)
        prog_gen.append(generation)
        print(f"Nowe najlepsze rozwiązanie znalezione w generacji {generation}")
        last_best_genome = best_genome
        last_best_fitness = best_fitness
        best_gen_video.append(best_genome)

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


# Wyświetlenie mapy z trasą
#print("Najlepsza trasa:", best_genome)
print("Odległość od celu:", last_best_fitness)

# Animation with Matplotlib
fig, ax = plt.subplots(figsize=(10, 10))

def update(frame):
    """Update function for animation."""
    ax.clear()
    best_gen = best_gen_video[frame]
    ax.imshow(mapa, cmap='gray_r')
    ax.plot(START[1], START[0], 'go', label="Start")  # Start
    ax.plot(GOAL[1], GOAL[0], 'ro', label="Goal")  # Goal
    position = START
    path_x, path_y = [position[1]], [position[0]]
    for command in best_gen:
        position = move_robot(position, command)
        path_x.append(position[1])
        path_y.append(position[0])
        if mapa[position] == 1:  # Collision with obstacle
            break
    ax.plot(path_x, path_y, 'b-', label="Robot Path")  # Path
    ax.legend()
    ax.set_title("Map with Obstacles and Robot Path")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    ax.set_xticks(np.arange(0, M, 1))
    ax.set_yticks(np.arange(0, N, 1))
    ax.grid()

# Create the animation
ani = FuncAnimation(
    fig, update, frames=len(best_gen_video),
    interval=300, repeat=False  # 500ms interval between frames
)

# Show the animation
plt.show()

# Save the animation using Pillow

writer = PillowWriter(fps=2)
ani.save('robot_path.gif', writer=writer)  # Save as gif file

progress = np.array(progress)
prog_gen = np.array(prog_gen)
prog_min = np.min(progress)
progress = int(np.sqrt(N*M)) * LEN_RATE - progress
plt.plot(prog_gen, progress)
plt.plot(prog_gen, np.zeros(len(progress)), 'r--')
plt.title("Postęp algorytmu genetycznego")
plt.xlabel("Generacja")
plt.ylabel("Odległość od celu")
plt.savefig('progress.png')
plt.show()


# Tworzenie mapowania ruchów na wartości liczbowe
move_to_number = {
    "straight": 0,
    "right": 1,
    "left": -1,
    "u-turn": 2
}

# Konwertowanie genomów na liczby dla x1 i x2
x1 = np.array([move_to_number.get(genome[0], 0) for genome in best_gen_video])  # Pierwszy gen lub 0
x2 = np.array([move_to_number.get(genome[1], 0) for genome in best_gen_video])  # Drugi gen lub 0

# Oś generacji
pokolenia = prog_gen

# Rysowanie wykresu 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1, x2, pokolenia, label='Najlepsze rozwiązanie w każdej generacji', color='k')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('Pokolenie')
ax.legend()
plt.title("Ewolucja parametrów w algorytmie genetycznym")

# Zapisanie wykresu jako plik parametry.png
plt.savefig('parametry.png')
plt.show()
