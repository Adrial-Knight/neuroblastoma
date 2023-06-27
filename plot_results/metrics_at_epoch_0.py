import os
import numpy as np
import json
import matplotlib.pyplot as plt

FIGURES = "../doc/LaTex/rapport/images/appendix/distribution"

def get_0_values(path: str, metric: str):
    # liste stockant les valeurs des epochs 0
    rand_train, rand_valid = [], []
    networks = os.listdir(path)
    for n in networks:
        cells = os.listdir(f"{path}/{n}")
        for c in cells:
            exps = os.listdir(f"{path}/{n}/{c}")
            for e in exps:
                with open(f"{path}/{n}/{c}/{e}/details.json", "r") as fd:
                    data = json.load(fd)
                    rand_train.append(data[f"train_{metric}"][0])
                    rand_valid.append(data[f"valid_{metric}"][0])
    return rand_train, rand_valid

def compute_mean_std(rand_list):
    res = {
        "mean": np.mean(np.array(rand_list)),
        "std":  np.std(np.array(rand_list))
    }
    return res

def main(path, metric):
    train_loss, valid_loss = get_0_values(path, metric)
    all_loss = train_loss + valid_loss

    # Affichage des résultats
    print(f"Results on {len(train_loss)} experiences.")
    print(f"{metric.capitalize()}:\n{compute_mean_std(all_loss)}")


    # Répartition
    plt.figure()
    plt.hist(valid_loss, bins=50, label="Validation", color="C1")
    plt.hist(train_loss, bins=50, label="Training", color="C0")
    plt.legend(title="Datasets")
    if metric == "accu": metric = "accuracy"
    plt.xlabel(f"{metric.capitalize()} at epoch 0")
    plt.ylabel("Nb experiences")
    # plt.title(f"Repartition of {metric} without training")
    plt.grid(True)


if __name__ == "__main__":
    path = "../backup"
    for metric in ["loss", "accu"]:
        main(path, metric)
        plt.savefig(f"{FIGURES}/{metric}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
        # plt.savefig(f"{FIGURES}/{metric}.png", format="png", bbox_inches="tight", pad_inches=0)
    # plt.show(block=True)
