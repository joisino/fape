import argparse

import numpy as np
from graphillion import setset

import fape

from joblib import Parallel, delayed

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--group_size', type=int, default=8)
parser.add_argument('--package_size', type=int, default=4)
parser.add_argument('--delta_proportionality', type=float, default=0.05)
parser.add_argument('--delta_envyfree', type=float, default=2 / 8 - 0.01)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--dataset', type=str, default='1m.npy')
parser.add_argument('--num_seed', type=int, default=10)
parser.add_argument('--n_jobs', type=int, default=-1)
args = parser.parse_args()


def process(
        seed,
        tau_prop,
        tau_envy,
        group_size,
        package_size,
        delta_proportionality,
        delta_envyfree,
        dataset):

    R_raw = np.load(dataset)
    m, n = R_raw.shape

    np.random.seed(seed)
    group = np.random.choice(np.arange(n), size=group_size, replace=False)

    R = R_raw[group]

    prop_satisfied = np.zeros(R.shape, dtype=bool)
    for i in range(group_size):
        threshold = np.sort(R[i])[int(n * (1 - delta_proportionality))]
        prop_satisfied[i] = R[i] >= threshold - 1e-9
    envy_satisfied = np.zeros(R.shape, dtype=bool)
    for j in range(n):
        threshold = np.sort(R[:, j])[int(group_size * (1 - delta_envyfree))]
        envy_satisfied[:, j] = R[:, j] >= threshold - 1e-9

    def scores(package):
        prop = (prop_satisfied[:, package].sum(1) >= 1).mean()
        envy = (envy_satisfied[:, package].sum(1) >= 1).mean()
        preference = ((R[:, package] - R.min()) / (R.max() - R.min())).mean()
        s = prop + envy + preference
        return s, prop, envy, preference

    setset.set_universe([i for i in range(n)])
    weight = {i: R[:, i].mean() for i in range(n)}

    res = fape.construct_zdd(R, package_size, tau_prop, 'proportionality', delta_proportionality)
    zdd_prop = setset(setset.loads(res))
    res = fape.construct_zdd(R, package_size, tau_envy, 'envyfree', delta_envyfree)
    zdd_envy = setset(setset.loads(res))
    zdd = zdd_prop & zdd_envy
    count = min(len(zdd), 100)
    res_top = []
    for package in zdd.max_iter(weight):
        if count == 0:
            break
        package = list(package)
        s, prop, envy, preference = scores(package)
        res_top.append((s, prop, envy, preference, tau_prop, tau_envy, package))
        count -= 1
    res_rand = None
    for package in zdd.rand_iter():
        package = list(package)
        s, prop, envy, preference = scores(package)
        res_rand = (s, prop, envy, preference, tau_prop, tau_envy, package)
        break
    return seed, res_top, res_rand


joblist = [
    (seed, tau_prop, tau_envy) for seed in range(args.num_seed) for tau_prop in range(1, 9) for tau_envy in range(1, 9)
]

print(joblist)

res = Parallel(n_jobs=args.n_jobs, verbose=10)(
    [delayed(process)(
        seed,
        tau_prop,
        tau_envy,
        args.group_size,
        args.package_size,
        args.delta_proportionality,
        args.delta_envyfree,
        args.dataset
    ) for seed, tau_prop, tau_envy in joblist])

data_backet = {seed: [] for seed in range(args.num_seed)}
data_backet_random = {}
for seed, res_top, res_rand in res:
    data_backet[seed] += res_top
    if res_rand:
        s, prop, envy, preference, tau_prop, tau_envy, package = res_rand
        data_backet_random[(seed, tau_prop, tau_envy)] = res_rand

props = [[] for i in range(4)]
envys = [[] for i in range(4)]
preferences = [[] for i in range(4)]
sums = [[] for i in range(4)]
for seed in range(args.num_seed):
    packages = sorted(data_backet[seed])
    for k, ind in enumerate([1, 10, 100]):
        res = packages[-ind]

        props[k].append(packages[-ind][1])
        envys[k].append(packages[-ind][2])
        preferences[k].append(packages[-ind][3])
        sums[k].append(packages[-ind][0])

    tau_prop = packages[-1][4]
    tau_envy = packages[-1][5]

    random_package = data_backet_random[(seed, tau_prop, tau_envy)]

    props[3].append(random_package[1])
    envys[3].append(random_package[2])
    preferences[3].append(random_package[3])
    sums[3].append(random_package[0])

for k in range(4):
    print(f'{k}', f'proportionality: {np.mean(props[k]):.3f} ± {np.std(props[k]):.3f},', f'envyfree: {np.mean(envys[k]):.3f} ± {np.std(envys[k]):.3f},',
          f'preference: {np.mean(preferences[k]):.3f} ± {np.std(preferences[k]):.3f},', f'total: {np.mean(sums[k]):.3f} ± {np.std(sums[k]):.3f}.')
