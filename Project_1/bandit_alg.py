import numpy as np
import matplotlib.pyplot as plt


# Bandit problem definition
class BernoulliBandit:
    def __init__(self, arms_probs):
        self.arms_probs = arms_probs
        self.optimal_arm = np.argmax(self.arms_probs)

    def pull(self, arm):
        return np.random.binomial(1, self.arms_probs[arm])


# ETC algorithms
def etc_optimal(bandit, n):
    delta = abs(bandit.arms_probs[0] - bandit.arms_probs[1])

    if delta == 0:
        k = n // 2
    else:
        k = int((4 / delta ** 2) * np.log(n * (delta ** 2) / 4))

    counts = [0, 0]
    values = [0.0, 0.0]

    for arm in range(2):
        for _ in range(k):
            reward = bandit.pull(arm)
            values[arm] += reward
            counts[arm] += 1

    best_arm = np.argmax(values)
    for _ in range(n - 2 * k):
        reward = bandit.pull(best_arm)
        values[best_arm] += reward
        counts[best_arm] += 1

    return counts, values


def etc_heuristic(bandit, n, k):
    counts = [0, 0]
    values = [0.0, 0.0]

    for arm in range(2):
        for _ in range(k):
            reward = bandit.pull(arm)
            values[arm] += reward
            counts[arm] += 1

    best_arm = np.argmax(values)
    for _ in range(n - 2 * k):
        reward = bandit.pull(best_arm)
        values[best_arm] += reward
        counts[best_arm] += 1

    return counts, values


# Simulation
n = 10000
k = 100
problems = {
    'P1': [0.8, 0.6],
    'P2': [0.8, 0.7],
    'P3': [0.55, 0.45],
    'P4': [0.5, 0.5]
}

for name, probs in problems.items():
    bandit = BernoulliBandit(probs)
    optimal_arm_played_all_runs = np.zeros((100, n))
    heuristic_arm_played_all_runs = np.zeros((100, n))

    regret_optimal_all_runs = np.zeros((100, n))
    regret_heuristic_all_runs = np.zeros((100, n))

    for run in range(100):
        counts_optimal, _ = etc_optimal(bandit, n)
        counts_heuristic, _ = etc_heuristic(bandit, n, k)

        optimal_arm_played_all_runs[run] = np.cumsum(
            [bandit.optimal_arm == np.argmax(counts_optimal[:t + 1]) for t in range(n)])
        heuristic_arm_played_all_runs[run] = np.cumsum(
            [bandit.optimal_arm == np.argmax(counts_heuristic[:t + 1]) for t in range(n)])

        regret_optimal_all_runs[run] = np.cumsum(probs[bandit.optimal_arm] * np.arange(1, n + 1)) - np.cumsum(
            [counts_optimal[0] * probs[0] + counts_optimal[1] * probs[1] for t in range(n)])
        regret_heuristic_all_runs[run] = np.cumsum(probs[bandit.optimal_arm] * np.arange(1, n + 1)) - np.cumsum(
            [counts_heuristic[0] * probs[0] + counts_heuristic[1] * probs[1] for t in range(n)])

    SE_optimal = np.std(optimal_arm_played_all_runs, axis=0) / np.sqrt(100)
    SE_heuristic = np.std(heuristic_arm_played_all_runs, axis=0) / np.sqrt(100)

    SE_regret_optimal = np.std(regret_optimal_all_runs, axis=0) / np.sqrt(100)
    SE_regret_heuristic = np.std(regret_heuristic_all_runs, axis=0) / np.sqrt(100)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.errorbar(range(n), np.mean(optimal_arm_played_all_runs, axis=0), yerr=SE_optimal, label='Optimal ETC')
    plt.errorbar(range(n), np.mean(heuristic_arm_played_all_runs, axis=0), yerr=SE_heuristic, label='Heuristic ETC',
                 linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Optimal Arm Played (%)')
    plt.title(f'{name} - Optimal Arm Played')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.errorbar(range(n), np.mean(regret_optimal_all_runs, axis=0), yerr=SE_regret_optimal, label='Optimal ETC')
    plt.errorbar(range(n), np.mean(regret_heuristic_all_runs, axis=0), yerr=SE_regret_heuristic, label='Heuristic ETC',
                 linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.title(f'{name} - Cumulative Regret')
    plt.legend()

    plt.tight_layout()
    plt.show()


k_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

print("Problem | k-value | Cumulative Regret at n")

for name, probs in problems.items():
    bandit = BernoulliBandit(probs)
    for k in k_values:
        regrets = []
        for _ in range(100):
            counts_heuristic, _ = etc_heuristic(bandit, n, k)
            regret_heuristic = np.cumsum(probs[bandit.optimal_arm] * np.arange(1, n + 1)) - np.cumsum(
                [counts_heuristic[0] * probs[0] + counts_heuristic[1] * probs[1] for t in range(n)])
            regrets.append(regret_heuristic[-1])  # Storing final regret after n rounds for each run

        avg_regret = np.mean(regrets)
        print(f"{name} | {k} | {avg_regret:.2f}")

print("\nExperiment finished!")