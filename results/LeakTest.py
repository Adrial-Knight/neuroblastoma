import numpy as np
import matplotlib.pyplot as plt
import os
import json

FIG_FOLDER = "../doc/presentation/experience"


def get_data(path: str, metric: str):
    experiences = os.listdir(path)
    data = {}
    for exp in experiences:
        exp_path  = f"{path}/{exp}"
        attempts  = os.listdir(exp_path)
        train_tab = [[] for _ in range(len(attempts))]
        valid_tab = [[] for _ in range(len(attempts))]
        for i, att in enumerate(attempts):
            json_path = f"{exp_path}/{att}/details.json"
            with open(json_path) as fd:
                content = json.load(fd)
            train_tab[i] = content[f"train_{metric}"]
            valid_tab[i] = content[f"valid_{metric}"]
        train_tab = np.array(train_tab)
        valid_tab = np.array(valid_tab)
        if metric == "accu":
            train_tab *= 100
            valid_tab *= 100
        data[exp] = {
                        "train_mean": np.mean(train_tab, axis=0),
                        "train_std":  np.std(train_tab,  axis=0),
                        "valid_mean": np.mean(valid_tab, axis=0),
                        "valid_std":  np.std(valid_tab,  axis=0)
                    }
    return data

def create_fig(data: dict, exp_num: list, metric: str):
    plt.figure(figsize=(9, 6))
    colors = ["C2", "orange", "C3"]
    for idx, num in enumerate(exp_num):
        exp = data[f"exp_{num}"]
        plt.plot(exp["valid_mean"], linestyle="-",  color=colors[idx], label=f"Exp {idx}")
        plt.fill_between(range(np.size(exp["valid_mean"])), exp["valid_mean"] - exp["valid_std"], exp["valid_mean"] + exp["valid_std"], alpha=0.3, color=colors[idx], label="_no_label_")
    if metric == "accu":
        plt.ylabel("Performance (%)", fontsize=15)
    else:
        plt.ylabel("Perte", fontsize=15)
    plt.xlabel("epochs", fontsize=15)
    plt.tick_params(axis="both", labelsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig(f"{FIG_FOLDER}/{metric}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)

if __name__ == "__main__":
    path = "../backup/LeakTest"
    exp_num = [0, 3, 4]
    for metric in ["accu", "loss"]:
        data = get_data(path, metric)
        create_fig(data, exp_num, metric)
