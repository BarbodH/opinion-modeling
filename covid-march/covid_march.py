print("Starting program...")

import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print("Imported necessary modules")


def covid_march():
    print("Initialization...")
    # initialization
    global quarantine
    random.seed(123)
    N = 20
    tmax = 365
    nruns = 1
    # variable associated with SIR (susceptible, infected, recovered)
    # probability to infect
    beta = 0.05
    # days to recover
    recovery_time = 14
    # probability to show symptoms
    epsilon = 1 / 20

    # one column for each opinion (1: for, -1: against)
    avrg_opinion_vs_t = np.zeros((tmax, 2))
    # one column for each disease state (S, I_A, I_S, R)
    avrg_disease_vs_t = np.zeros((tmax, 4))

    for run in range(1, nruns + 1):
        print(run)
        opinion_network = nx.barabasi_albert_graph(n=N, m=8)
        disease_network = nx.barabasi_albert_graph(n=N, m=3)

        t = 0
        opinion = set_initial_opinion(N)
        self_timer = np.zeros((N, 1))
        disease = np.zeros((N, 1))
        disease[np.random.randint(N)] = 1

        while t < tmax:
            p = increase_opinion(t)
            q = decrease_opinion(t)
            opinion_aux = update_opinions(
                disease, opinion, N, opinion_network, disease_network, p, q
            )
            if t <= 15 or t >= 120:
                quarantine = np.zeros((N, 1))
            [disease_aux, self_timer] = update_disease(
                disease,
                opinion,
                N,
                disease_network,
                self_timer,
                beta,
                recovery_time,
                epsilon,
            )
            opinion = opinion_aux
            disease = disease_aux
            t += 1

            avrg_opinion_vs_t[t, :] += count_op(opinion)
            avrg_disease_vs_t[t, :] += count_di(disease)

        # divide by total runs for average and by N for normalization
        avrg_opinion_vs_t = avrg_opinion_vs_t / (N * nruns)
        avrg_disease_vs_t = avrg_disease_vs_t / (N * nruns)

        data = avrg_disease_vs_t[:, 1] + avrg_disease_vs_t[:, 2]
        pks, locs = find_peaks(data, prominence=0.01)

        # plot results
        time = np.arange(1, tmax + 1).reshape(-1, 1)

        # Figure 1
        h1 = plt.figure(1)
        plt.rc("font", size=18)
        plt.plot(time, avrg_opinion_vs_t, linewidth=3)
        plt.xlabel("Time (Days)")
        plt.ylabel("Opinion")
        plt.tick_params(direction="out", width=3)
        plt.legend(["-1", "+1"])

        # Figure 2
        h2 = plt.figure(2)
        plt.plot(time, avrg_disease_vs_t[:, 1] + avrg_disease_vs_t[:, 2], linewidth=3)
        plt.plot(time[locs], pks, "r*")
        plt.xlabel("Time (Days)")
        plt.ylabel("Population")
        plt.tick_params(direction="out", width=3)
        plt.legend(["S", "E", "I", "R"])

        plt.show()


def increase_opinion(t):
    # probability to increase opinion (move right)
    p0 = 0.01
    p = p0 * math.exp(-p0 * t)
    return p


def decrease_opinion(t):
    # probability to decrease opinion (move left)
    mu = 90
    s = 50
    q = 0.02 / (0.02 + math.exp(-(t - mu) / s))
    return q


def set_initial_opinion(N):
    op_out = np.random.choice([-1, 1], size=(N, 1))
    return op_out


def update_opinions(
    old_disease, old_opinion, N, opinion_network, disease_network, p, q
):
    global quarantine

    quarantine = np.zeros((N, 1))
    new_opinion = np.zeros((N, 1))
    c = 0.5
    eta = 0.1
    beta1 = 1
    nu = -0.1

    for current_node_index in range(0, N):
        print(old_opinion)
        neighbors_opinion = get_neighbors(opinion_network, current_node_index)
        neighbors_disease = get_neighbors(disease_network, current_node_index)
        number_infected = 0

        # calculate the number of infected nodes around the current node
        for neighbor_index in neighbors_disease:
            if old_disease[neighbor_index] == 2:
                number_infected = number_infected + 1

        Itot = count_di(old_disease)[2]
        Pne = -1 * number_infected
        Ppo = nu
        x = Ppo - Pne
        Gamma = 1 / (1 + math.exp(-beta1 * x))

        random_index = np.random.permutation(len(neighbors_opinion))[0]
        random_neighbor = neighbors_opinion[random_index]

        # 16 total combinations
        if old_opinion[current_node_index] == old_opinion[random_neighbor]:
            if old_opinion[current_node_index] < 0 and random.random() < Gamma:
                new_opinion[current_node_index] = 1
            elif old_opinion[current_node_index] > 0 and random.random() < (1 - Gamma):
                new_opinion[current_node_index] = -1
            else:
                new_opinion[current_node_index] = old_opinion[current_node_index]

        elif old_opinion[current_node_index] < old_opinion[random_neighbor]:
            if random.random() < (0.5 * Gamma + 0.5 * p):
                new_opinion[current_node_index] = 1
            else:
                new_opinion[current_node_index] = old_opinion[current_node_index]

        elif old_opinion[current_node_index] > old_opinion[random_neighbor]:
            if random.random() < (0.5 * (1 - Gamma) + 0.5 * q):
                new_opinion[current_node_index] = -1
            else:
                new_opinion[current_node_index] = old_opinion[current_node_index]

    quarantine[new_opinion == 1] = 1


def next(opinion):
    if opinion == -1:
        return 1
    if opinion == 1:
        return 1
    # redundant?


def prev(opinion):
    if opinion == -1:
        return -1
    if opinion == 1:
        return -1
    # redundant?


# *** indexing might cause trouble (starting from 0 instead of 1)
def get_neighbors(graph, node_index):
    return list(graph.neighbors(node_index))


def count_di(disease):
    total_count = [
        np.sum(disease == 0),
        np.sum(disease == 1),
        np.sum(disease == 2),
        np.sum(disease == 3),
    ]
    return total_count


def count_op(opinion):
    total_count = [np.sum(opinion == -1), np.sum(opinion == 1)]
    return total_count


def update_disease(
    old_state, opinion, N, disease_network, self_timer, beta, recovery_time, epsilon
):
    global quarantine
    new_state = old_state

    list_of_infected = np.where(old_state == 2)[0]
    list_of_exposed = np.where(old_state == 1)[0]

    for index in range(0, len(list_of_infected)):
        i = list_of_infected[index]
        self_timer[i] += 1
        if self_timer[i] == recovery_time:
            new_state[i] = 3

        if quarantine[i] == 0:
            infect_neighbor(i, disease_network, beta, old_state, new_state, self_timer)

        if random.random() < epsilon:
            new_state[i] = 2
            self_timer[i] = 0
            opinion[i] = 2
            quarantine[i] = 1

    return [new_state, self_timer]


def infect_neighbor(node, disease_network, beta, old_state, new_state, self_timer):
    neighbors = get_neighbors(node, disease_network)
    for j in range(0, len(neighbors)):
        neigh = neighbors[j]
        r = random.random()
        if r < beta and quarantine[neigh] == 0 and old_state[neigh] == 0:
            new_state[neigh] = 1
            self_timer[neigh] = 0


covid_march()
