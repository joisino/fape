# Enumerating Fair Packages for Group Recommendations (WSDM 2022)

We proposed an efficient method to enumerate *all* fair packages with respect to envy-freeness and proportionality.

Paper: https://arxiv.org/abs/2105.14423

## 💿 Installation

```
$ pip install fape
```

## 💡 How to Use

This package is compatible with [graphillion](https://github.com/takemaru/graphillion) and [SAPPOROBDD](http://www.lab2.kuis.kyoto-u.ac.jp/minato/SAPPOROBDD/). Please install graphillion via

```
$ pip install graphillion
```

### Basic Usage

`fape.construct_zdd` constructs ZDD and outputs a ZDD string. 

```python
import fape
import numpy as np
from graphillion import setset

m = 3
n = 4
R = np.array([
    [10.0, 20.0, 5.0, 5.0],
    [10.0, 5.0, 5.0, 30.0],
    [10.0, 20.0, 5.0, 5.0],
]) # 3 members x 4 itmes

zdd_string = fape.construct_zdd(
    R,
    package_size=2,
    tau=3,
    criterion='proportionality',
    delta=0.5
)
setset.set_universe([i for i in range(n)])
ss = setset(setset.loads(zdd_string))
for r in ss:
    print(r)
# {0, 1}
# {0, 2}
# {0, 3}
# {1, 3}

weight = {
    i: R[:, i].mean() for i in range(n)
}
for r in ss.max_iter(weight):
    print(r)
    break
# {1, 3}
```

In this example, `delta = 0.5` means that each member likes the items with top talf ratings. Namely,

* member 0 likes items 0 and 1,
* member 1 likes items 0 and 3, and
* member 2 likes items 0 and 1 (zero-indexed).

`tau = 3` means that three members (i.e., all members) should be satisfied. `fape.construct_zdd` enumerates such packages. There are qualified four packages, {0, 1}, {0, 2}, {0, 3}, and {1, 3}.

`weight` is a dictionary that stores the average preference of each item. `max_iter` iterates packages in the descreasing order of weights. In this example, package {1, 3} has the largest weight.

A `graphillion.setset.setset` supports various operations, including union (`|`) and intersection (`&`). Please refer to the document of [graphillion](https://github.com/takemaru/graphillion) for more details.

## 📝 Results

|Dataset<br>Metric|MovieLens<br>Proportionality|MovieLens<br>Envyfreeness|MovieLens<br>Preference|MovieLens<br>TotalScore|Amazon<br>Proportionality|Amazon<br>Envyfreeness|Amazon<br>Preference|Amazon<br>TotalScore|
|---|---|---|---|---|---|---|---|---|
|Averanking| **1.000 ± 0.000** | 0.725 ± 0.156 | **0.911 ± 0.027** | 2.636 ± 0.171 | **1.000 ± 0.000** | 0.500 ± 0.125 | **0.939 ± 0.012** | 2.439 ± 0.126 |
|LMRanking| 0.988 ± 0.037 | 0.588 ± 0.168 | 0.876 ± 0.036 | 2.451 ± 0.188 | 0.912 ± 0.263 | 0.425 ± 0.139 | 0.924 ± 0.031 | 2.261 ± 0.373 |
|GreedyVar| 0.912 ± 0.080 | 0.750 ± 0.112 | 0.812 ± 0.035 | 2.474 ± 0.157 | 0.787 ± 0.202 | 0.637 ± 0.088 | 0.859 ± 0.031 | 2.284 ± 0.287 |
|GreedyLM| 0.950 ± 0.061 | 0.775 ± 0.109 | 0.813 ± 0.036 | 2.538 ± 0.155 | 0.662 ± 0.159 | 0.600 ± 0.094 | 0.853 ± 0.031 | 2.115 ± 0.249 |
|GFAR| 0.950 ± 0.061 | 0.762 ± 0.104 | 0.812 ± 0.038 | 2.525 ± 0.154 | 0.762 ± 0.142 | 0.650 ± 0.075 | 0.871 ± 0.025 | 2.284 ± 0.219 |
|SPGreedy| **1.000 ± 0.000** | 0.525 ± 0.156 | 0.851 ± 0.041 | 2.376 ± 0.167 | 1.000 ± 0.000 | 0.375 ± 0.079 | 0.867 ± 0.015, | 2.242 ± 0.085 |
|EFGreedy| 0.925 ± 0.127 | **1.000 ± 0.000** | 0.792 ± 0.053 | 2.717 ± 0.165 | 0.750 ± 0.244 | 0.838 ± 0.080 | 0.854 ± 0.027 | 2.441 ± 0.302 |
|FAPE(ours, exact)| **1.000 ± 0.000** | **1.000 ± 0.000** | 0.888 ± 0.037 | **2.888 ± 0.037** | **1.000 ± 0.000** | **0.912 ± 0.057** | 0.913 ± 0.020 | **2.825 ± 0.064** |
|FAPE(ours, 10th)| **1.000 ± 0.000** | **1.000 ± 0.000** | 0.887 ± 0.037 | 2.887 ± 0.037 | **1.000 ± 0.000** | **0.912 ± 0.057** | 0.911 ± 0.020 | 2.824 ± 0.064 |
|FAPE(ours, 100th)| **1.000 ± 0.000** | **1.000 ± 0.000** | 0.881 ± 0.040 | 2.881 ± 0.040 | **1.000 ± 0.000** | 0.900 ± 0.050 | 0.905 ± 0.025 | 2.805 ± 0.058 |
|FAPE(ours, random)| **1.000 ± 0.000** | **1.000 ± 0.000** | 0.720 ± 0.044 | 2.720 ± 0.044 | **1.000 ± 0.000** | 0.912 ± 0.057 | 0.878 ± 0.023 | 2.790 ± 0.069 |

Please refer to the paper for more details.

## 🧪 How to Reproduce

### Dependency of python scripts

Please install [graphillion](https://github.com/takemaru/graphillion) and [surprise](http://surpriselib.com/) via `pip install graphillion scikit-surprise`

### Data

You can download and preprocess data by the following command. It may take time. Please use only MovieLens 100k/1m if it takes too much time.

```
$ bash download.sh
```

`100k.npy`, `1m.npy`, `10m.npy`, `20m.npy`, and `25m.npy` are variants of the MovieLens dataset. `home_and_kitchen.npy` is the Amazon dataset.

### Evaluation Scripts

* `speed.py` measures the speed of FAPE (Section 4.2).
* `evaluate_baselines.py` evaluates the baseline methods (Section 4.3).
* `evaluate_ours.py` evaluates FAPE (Section 4.3).

## Citation

```
@inproceedings{sato2022enumerating,
  author    = {Ryoma Sato},
  title     = {Enumerating Fair Packages for Group Recommendations},
  booktitle = {Proceedings of the Fifteenth {ACM} International Conference on Web Search and Data Mining, {WSDM}},
  year      = {2022},
}
```