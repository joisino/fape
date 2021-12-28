import time

import argparse

import numpy as np
from graphillion import setset
import fape

np.random.seed(0)


def create_group(R, t, th, group_size, seed):
    m, n = R.shape
    np.random.seed(seed)
    if t == 'random':
        group = np.random.choice(np.arange(m), size=group_size, replace=False)
    elif t == 'similar':
        coef = np.corrcoef(R)
        while True:
            group = [np.random.randint(m)]
            while len(group) < group_size:
                ind = np.where(coef[group].min(0) >= th)[0].tolist()
                ind = list(set(ind) - set(group))
                if len(ind) == 0:
                    break
                group.append(np.random.choice(ind))
            if len(group) == group_size:
                break
    elif t == 'divergent':
        coef = np.corrcoef(R)
        while True:
            group = [np.random.randint(m)]
            while len(group) < group_size:
                ind = np.where(coef[group].min(0) <= th)[0].tolist()
                ind = list(set(ind) - set(group))
                if len(ind) == 0:
                    break
                group.append(np.random.choice(ind))
            if len(group) == group_size:
                break
    return np.array(group)


def measure(R, package_size, tau, fairness_type, delta):
    start = time.time()
    res = fape.construct_zdd(R, package_size, tau, fairness_type, delta)
    t = time.time() - start

    setset.set_universe([i for i in range(R.shape[1])])
    ss = setset.loads(res)

    s = ss.len()
    return t, s


parser = argparse.ArgumentParser()
parser.add_argument('--group_size', type=int, default=8)
parser.add_argument('--tau', type=int, default=8)
parser.add_argument('--package_size', type=int, default=4)
parser.add_argument('--delta', type=float, default=0.05)
parser.add_argument('--group_type', type=str, default='random', choices=['random', 'similar', 'divergent'])
parser.add_argument('--group_threshold', type=float, default=0)
parser.add_argument('--dataset', type=str, default='1m.npy')
parser.add_argument('--fairness_type', type=str, default='proportionality', choices=['proportionality', 'envyfree'])
parser.add_argument('--repeat', type=int, default=10)
args = parser.parse_args()

group_size = args.group_size
tau = args.tau
package_size = args.package_size
delta = args.delta

group_type = args.group_type
group_threshold = args.group_threshold

dataset = args.dataset

fairness_type = args.fairness_type

print('# Package Size')
for package_size_cur in range(2, 9):
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, group_type, group_threshold, group_size, seed)
        t, s = measure(R_raw[group], package_size_cur, tau, fairness_type, delta)
        times.append(t)
        sizes.append(s)

    print(f'{package_size_cur}. {dataset}, group_size: {group_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')

print('# Group Size')
for group_size_cur in [2, 4, 8, 12, 16]:
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, group_type, group_threshold, group_size_cur, seed)
        t, s = measure(R_raw[group], package_size, group_size_cur, fairness_type, delta)
        times.append(t)
        sizes.append(s)

    print(f'{group_size_cur}. {dataset}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')

print('# Delta')
for delta_cur in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, group_type, group_threshold, group_size, seed)
        t, s = measure(R_raw[group], package_size, tau, fairness_type, delta_cur)
        times.append(t)
        sizes.append(s)

    print(f'{delta_cur}. {dataset}, group_size: {group_size}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')

print('# Delta (envyfree)')
for delta_cur in [i / group_size - 0.01 for i in range(1, group_size + 1)]:
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, group_type, group_threshold, group_size, seed)
        t, s = measure(R_raw[group], package_size, tau, 'envyfree', delta_cur)
        times.append(t)
        sizes.append(s)

    print(f'{delta_cur}. {dataset}, group_size: {group_size}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')

print('# Tau')
for tau_cur in range(1, 9):
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, group_type, group_threshold, group_size, seed)
        t, s = measure(R_raw[group], package_size, tau_cur, fairness_type, delta)
        times.append(t)
        sizes.append(s)

    print(f'{tau_cur}. {dataset}, group_size: {group_size}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')


print('# Group Threshold (similar)')
for group_threshold_cur in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, 'similar', group_threshold_cur, group_size, seed)
        t, s = measure(R_raw[group], package_size, tau, fairness_type, delta)
        times.append(t)
        sizes.append(s)

    print(f'{group_threshold_cur}. {dataset}, group_size: {group_size}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')

print('# Group Threshold (divergent)')
for group_threshold_cur in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    R_raw = np.load(dataset)

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, 'divergent', group_threshold_cur, group_size, seed)
        t, s = measure(R_raw[group], package_size, tau, fairness_type, delta)
        times.append(t)
        sizes.append(s)

    print(f'{group_threshold_cur}. {dataset}, group_size: {group_size}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')


print('# Dataset Size')
for dataset_cur in ['100k.npy', '1m.npy', '10m.npy', '20m.npy', '25m.npy']:
    R_raw = np.load(dataset_cur)
    m, n = R_raw.shape

    times = []
    sizes = []
    for seed in range(args.repeat):
        group = create_group(R_raw, group_type, group_threshold, group_size, seed)
        t, s = measure(R_raw[group], package_size, tau, fairness_type, delta)
        times.append(t)
        sizes.append(s)

    print(f'{dataset_cur}. user: {m}, item: {n}, group_size: {group_size}, package_size: {package_size}, time: {np.mean(times)} ± {np.std(times)}, size: {np.mean(sizes)} ± {np.std(sizes)}')
