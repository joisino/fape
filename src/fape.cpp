#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

using ll = long long int;

string id_to_str(int x){
  if(x == 0){
    return "B";
  } else if(x == 1){
    return "T";
  }
  return to_string(x);
}

string construct_zdd(pybind11::array_t<double> x, int package_size, int tau, string criterion, float delta){
    const auto &buff_info = x.request();
    const auto &shape = buff_info.shape;
    int m = shape[0];
    int n = shape[1];
    vector<vector<float>> R(m, vector<float>(n));
    for(int u = 0; u < m; u++){
        for(int i = 0; i < n; i++){
            R[u][i] = *x.data(u, i);
        }
    }

    vector<int> S(n, 0);
    if(criterion == "proportionality"){
        for(int u = 0; u < m; u++){
            vector<float> v = R[u];
            sort(v.begin(), v.end());
            float th = v[int(n * (1 - delta))];
            for(int i = 0; i < n; i++){
                if(R[u][i] >= th){
                    S[i] |= 1 << u;
                }
            }
        }
    } else if(criterion == "envyfree"){
        for(int i = 0; i < n; i++){
            vector<float> v(0);
            for(int u = 0; u < m; u++){
                v.push_back(R[u][i]);
            }
            sort(v.begin(), v.end());
            float th = v[int(m * (1 - delta))];
            for(int u = 0; u < m; u++){
                if(R[u][i] >= th){
                    S[i] |= 1 << u;
                }
            }
        }
    } else {
        assert(false);
    }

    ll max_node = n * (1 << m) * (package_size + 1) + 2;
    unordered_map<ll, int> p;
    vector<vector<vector<int>>> s(n, vector<vector<int>>(package_size+1, vector<int>(1<<m)));
    int var_cnt = 2;
    vector<tuple<int, int, string, string>> ans;
    int root = 0;
    for(int i = n-1; i >= 0; i--){
        for(int h = 0; h < (1 << m); h++){
            int hp = h | S[i];
            for(int k = 0; k <= package_size; k++){
                int hi, lo;
                if(i == n - 1){
                    if(k == package_size - 1 && __builtin_popcount(hp) >= tau){
                        hi = 1;
                    } else {
                        hi = 0;
                    }
                    if(k == package_size && __builtin_popcount(h) >= tau){
                        lo = 1;
                    } else {
                        lo = 0;
                    }
                } else if(k == package_size){
                    hi = 0;
                    lo = s[i+1][k][h];
                } else {
                    hi = s[i+1][k+1][hp];
                    lo = s[i+1][k][h];
                }
                ll cur = i * max_node * max_node + hi * max_node + lo;
                if(hi == 0){
                    s[i][k][h] = lo;
                } else if(p.find(cur) != p.end()){
                    s[i][k][h] = p[cur];
                } else {
                    p[cur] = var_cnt;
                    s[i][k][h] = var_cnt;
                    ans.emplace_back(var_cnt, i + 1, id_to_str(lo), id_to_str(hi));
                    var_cnt++;
                    if(h == 0 && k == 0){
                        root = ans.size() - 1;
                    }
                }
            }
        }
    }
    string zdd_string = "";
    for(int i = 0; i <= root; i++){
        int a, b;
        string c, d;
        tie(a, b, c, d) = ans[i];
        zdd_string += to_string(a) + " " + to_string(b) + " " + c + " " + d + "\n";
    }
    zdd_string += ".\n";
    return zdd_string;
}


PYBIND11_MODULE(fape, m) {
    m.doc() = "Fair package enuemeration for group recommendations";
    m.def(
        "construct_zdd",
        &construct_zdd,
        pybind11::arg("R"),
        pybind11::arg("package_size"),
        pybind11::arg("tau"),
        pybind11::arg("criterion"),
        pybind11::arg("delta"),
        "Construct a ZDD that represents all fair packages.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "R : numpy.array\n"
        "    Rating matrix with size (m, n).\n"
        "    m is the number of users.\n"
        "    n is the number of items.\n"
        "    R[i, j] indicates how much user i likes item j.\n"
        "\n"
        "package_size: int\n"
        "    Size of packages.\n"
        "\n"
        "tau: int\n"
        "    Threshold of the fairness value.\n"
        "    Packages with the fairness value greater than or equal to this value will be output.\n"
        "    Note that the fairness value is represented by the number of satisfied members (int),\n"
        "    NOT the ratio of satisfied members (float).\n"
        "\n"
        "criterion: {'proportionality', 'envyfree'}\n"
        "    Criterion of measure fairness.\n"
        "\n"
        "delta: float\n"
        "    Threshold used in the definition of fairness.\n"
        "    This values should be in [0, 1].\n"
        "    In proportionality, this values specifies the ratio of items each user likes.\n"
        "    In envyfreeness, this values specifies the ratio of members who is envyfree for an item.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "zdd_string: string\n"
        "    ZDD that represents all fair packages.\n"
        "    The format is compatible with Graphillion and SAPPOROBDD.\n"
    );
}
