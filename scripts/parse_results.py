import collections
import pickle
import matplotlib.pyplot as plt
import sys

models = ["baseline", "ls"]
methods = ["sample", "sample_ts", "greedy", "explore"]
colors = ['blue', 'orange', 'green', 'red']
lines = ['solid', 'dashed']

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

def merge_results(fnames):
    results_0 = read_results(fnames[0])
    for fname in fnames[1:]:
        results_i = read_results(fname)
        for model, method in zip(models, methods):
            outs_0 = results_0[model][method]
            outs_i = results_i[model][method]
            merged_results = list()
            for k in range(4):
                assert outs_0[k][0] == outs_i[k][0]
                merged_results.append((outs_0[k][0], 0.5*(outs_0[k][1] + outs_i[k][1])))
            results_0[model][method] = merged_results
    return results_0



def plot_results(results):
    for i, model in enumerate(models):
        for j, method in enumerate(methods):
            print((model, method))
            print(results[model][method])
            results_sr = [x[1] for x in results[model][method]]
            results_k = [x[0] for x in results[model][method]]
            plt.plot(results_k, results_sr, label=f"{model}, {method}", color=colors[j], linestyle=lines[i])
    plt.legend(fontsize='xx-large')
    plt.title("Transformer Performance on Test Tasks", fontsize='xx-large')
    plt.xlabel("Number of subgoals", fontsize='xx-large')
    plt.ylabel("Success Rate", fontsize='xx-large')
    plt.xticks(ticks=[1, 2, 3, 4])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "new":
        make_empty_results("results.pkl")
    elif len(sys.argv) >= 2:
        results = merge_results(sys.argv[1:])
        plot_results(results)
    else:
        results = read_results("results.pkl")
        plot_results(results)

