import argparse

import numpy as np

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--group_size', type=int, default=8)
parser.add_argument('--package_size', type=int, default=4)
parser.add_argument('--delta_proportionality', type=float, default=0.05)
parser.add_argument('--delta_envyfree', type=float, default=2 / 8 - 0.01)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--dataset', type=str, default='1m.npy')
args = parser.parse_args()

group_size = args.group_size
package_size = args.package_size
delta_proportionality = args.delta_proportionality
delta_envyfree = args.delta_envyfree

lambda_greedy = args.lam

dataset = args.dataset

R_raw = np.load(dataset)
m, n = R_raw.shape


for algorithm in ['AveRanking', 'LMRanking', 'GreedyVar', 'GreedyLM', 'GFAR', 'SPGreedy', 'EFGreedy']:
    props = []
    envys = []
    preferences = []
    sums = []
    for seed in range(10):
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

        borda = np.zeros((group_size, n))
        borda[np.arange(group_size).repeat(package_size), np.argsort(R, 1)[:, -package_size:].reshape(-1)] = np.arange(package_size)[None, :].repeat(group_size, 0).reshape(-1)
        borda /= package_size - 1

        res = []
        if algorithm == 'AveRanking':
            res = R.sum(0).argsort()[-package_size:].tolist()
        elif algorithm == 'LMRanking':
            res = R.min(0).argsort()[-package_size:].tolist()
        elif algorithm == 'GreedyVar':
            for i in range(package_size):
                sc = np.ones(n) * (-1e18)
                for j in range(n):
                    if j in res:
                        continue
                    nres = res + [j]
                    sc[j] = lambda_greedy * borda[:, nres].mean() + (1 - lambda_greedy) * (1 - borda[:, nres].mean(1).var())
                res.append(np.argmax(sc))
        elif algorithm == 'GreedyLM':
            for i in range(package_size):
                sc = np.ones(n) * (-1e18)
                for j in range(n):
                    if j in res:
                        continue
                    nres = res + [j]
                    sc[j] = lambda_greedy * borda[:, nres].mean() + (1 - lambda_greedy) * borda[:, nres].mean(1).min()
                res.append(np.argmax(sc))
        elif algorithm == 'GFAR':
            p = borda / borda.sum(1, keepdims=True)
            res = []
            cur = np.ones(group_size)
            for i in range(package_size):
                s = (p * cur.reshape(group_size, 1)).sum(0)
                s[res] = -1e18
                c = np.argmax(s)
                res.append(c)
                cur *= 1 - p[:, c]
        elif algorithm == 'SPGreedy':
            remain = set(range(group_size))
            for i in range(package_size):
                cur = prop_satisfied[list(remain)].sum(0).argmax()
                res.append(cur)
                remain -= set(np.where(prop_satisfied[:, cur])[0].tolist())
        elif algorithm == 'EFGreedy':
            remain = set(range(group_size))
            for i in range(package_size):
                cur = envy_satisfied[list(remain)].sum(0).argmax()
                res.append(cur)
                remain -= set(np.where(envy_satisfied[:, cur])[0].tolist())

        prop = (prop_satisfied[:, res].sum(1) >= 1).mean()
        envy = (envy_satisfied[:, res].sum(1) >= 1).mean()
        preference = ((R[:, res] - R.min()) / (R.max() - R.min())).mean()
        s = prop + envy + preference

        props.append(prop)
        envys.append(envy)
        preferences.append(preference)
        sums.append(s)
    print(f'{algorithm:20}', f'proportionality: {np.mean(props):.3f} ± {np.std(props):.3f},', f'envyfree: {np.mean(envys):.3f} ± {np.std(envys):.3f},',
          f'preference: {np.mean(preferences):.3f} ± {np.std(preferences):.3f},', f'total: {np.mean(sums):.3f} ± {np.std(sums):.3f}.')
