import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as clr

FIGURES = "../doc/LaTex/rapport/images/appendix/distribution"

def get_values(path: str, dataset: str):
    # liste stockant les valeurs des epochs
    accu, loss = [], []
    networks = os.listdir(path)
    for n in networks:
        cells = os.listdir(f"{path}/{n}")
        for c in cells:
            exps = os.listdir(f"{path}/{n}/{c}")
            for e in exps:
                with open(f"{path}/{n}/{c}/{e}/details.json", "r") as fd:
                    data = json.load(fd)
                    accu += data[f"{dataset}_accu"]
                    loss += data[f"{dataset}_loss"]
    loss = np.array(loss)
    accu = np.array(accu) * 100
    return accu, loss

def display(accu, loss):
    min_accu = min(accu)
    max_loss = max(loss)
    rand_loss = 0.7
    rand_accu = 50
    threshold = 0  # 0: ne masque aucune bin

    # Calcul des bins hexagonaux
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(loss, accu, gridsize=100, cmap="inferno", norm=clr.LogNorm())

    # Mise à jour des compteurs pour les bins masqués
    counts = hb.get_array().data
    masked_counts = np.ma.masked_where(counts < threshold, counts)
    masked_counts[masked_counts.mask] = 0
    hb.set_array(masked_counts)

    fig.colorbar(hb).set_label("Density")
    plt.xlabel("Binary Cross Entropy")
    plt.ylabel("Accuracy (%)")
    plt.xlim([0, 0.8])
    plt.ylim([40, 100])
    plt.savefig(f"{FIGURES}/accu_vs_loss.pdf", format="pdf", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    accu, loss = get_values(path="../backup", dataset="train")
    print(f"Number of point: {len(accu)}")
    display(accu, loss)
