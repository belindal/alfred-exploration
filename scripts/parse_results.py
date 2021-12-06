import collections
import pickle
import matplotlib.pyplot as plt

models = ["baseline", "ls"]
methods = ["sample", "sample_ts", "greedy", "explore"]

def make_empty_results(fname):
    results = collections.defaultdict(dict)
    for model in models:
        for method in methods:
            results[model][method] = []
    pickle.dump(results, open(fname, "wb"))

def read_results(fname):
    results = pickle.load(open(fname, "rb"))
    return results

def write_result(fname, model, method, result):
    # result: (k subgoals, sr)
    results = read_results(fname)
    results[model][method].append(result)
    pickle.dump(results, open(fname, "wb"))

def plot_results():
    results = read_results("results.pkl")
    for model in models:
        for method in methods:
            print(results[model][method])
            results_sr = [x[1] for x in results[model][method]]
            results_k = [x[0] for x in results[model][method]]
            plt.plot(results_k, results_sr, label=f"{model}, {method}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    make_empty_results("results.pkl")
    #plot_results()

