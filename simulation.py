import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from simulation_utils import *
from individual import Individual


def simulate_epidemic():
    # initialization
    random.seed(123)
    N = 50
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
        print("Run:", run)
        opinion_network = nx.barabasi_albert_graph(n=N, m=8)
        disease_network = nx.barabasi_albert_graph(n=N, m=3)
        t = 0

        # generate the individuals in the network
        individuals = list()
        for i in range(0, N):
            individuals.append(Individual(i))
        individuals[random.randint(0, N)].disease = 1

        while t < tmax:
            p = increase_opinion(t)
            q = decrease_opinion(t)

            update_opinions(individuals, opinion_network, disease_network, p, q)

            # liberate the quarantine if necessary
            if t <= 15 or t >= 120:
                for individual in individuals:
                    individual.quarantine = 0

            update_disease(individuals, disease_network, beta, recovery_time, epsilon)

            t += 1

            avrg_opinion_vs_t[t - 1, :] += count_opinion_states(individuals)
            avrg_disease_vs_t[t - 1, :] += count_disease_states(individuals)

        # divide by total runs for average and by N for normalization
        avrg_opinion_vs_t = avrg_opinion_vs_t / (N * nruns)
        avrg_disease_vs_t = avrg_disease_vs_t / (N * nruns)

        data = avrg_disease_vs_t[:, 1] + avrg_disease_vs_t[:, 2]
        pks, locs = find_peaks(data, prominence=0.01)

        # plot results
        time = np.arange(1, tmax + 1).reshape(-1, 1)

        # Figure 1
        plt.figure(1)
        plt.rc("font", size=18)
        plt.plot(time, avrg_opinion_vs_t, linewidth=2)
        plt.xlabel("Time (Days)")
        plt.ylabel("Opinion")
        plt.tick_params(direction="out", width=3)
        plt.legend(["against", "for"])

        # Figure 2
        plt.figure(2)
        plt.plot(time, avrg_disease_vs_t, linewidth=2)
        # plt.plot(time[locs], pks, "r*")
        plt.xlabel("Time (Days)")
        plt.ylabel("Population")
        plt.tick_params(direction="out", width=3)
        plt.legend(["S", "E", "I", "R"])

        plt.show()


simulate_epidemic()
