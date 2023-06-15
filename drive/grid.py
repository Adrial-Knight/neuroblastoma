import matplotlib
matplotlib.use("Agg")

import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from tqdm import tqdm

try:
    from . import goolgeapiclient_wrap as Gdrive
except ImportError:
    import goolgeapiclient_wrap as Gdrive

SKIP_FOLDER = ["__Summary__"]

def main(drive, model):
    grid, active = make_grid(drive, model)
    display_grid(grid, active, model)

def make_grid(drive, model):
    root_path = f"Stage_Bilbao_Neuroblastoma/G_Collab/backup/{model}"
    root_id   = Gdrive.get_id_from_path(drive, root_path)
    title_list, id_list = Gdrive.list_from_id(drive, root_id)

    # suppression des dossiers ne faisant pas partis de la grille de recherche
    for skip in SKIP_FOLDER:
        if skip in title_list:
            index = title_list.index(skip)
            title_list.remove(skip)
            id_list.pop(index)

    grid   = {}
    active = {}
    for title, id in tqdm(zip(title_list, id_list), total=len(id_list)):
        if title in SKIP_FOLDER: continue
        exp_list, exp_id_list = Gdrive.list_from_id(drive, id)
        lr, batch = title.split("_")
        value = len(exp_list)
        if "tmp" in exp_list: value -= 1
        _, files = Gdrive.list_from_id(drive, exp_id_list[0])
        if files:
            infos = Gdrive.get_file_infos_from_id(drive, files[0], utc_offset=2)
            now = time.time()
            is_running = abs(now - infos["modifiedTimeSec"]) < 120
        else:
            is_running = True
        grid[(lr, batch)] = value
        active[(lr, batch)] = is_running
    return grid, active

def display_grid(grid, active, model):
    # Récupérer les coordonnées et les valeurs
    coordinates = list(grid.keys())
    values = list(grid.values())

    # Convertir les coordonnées en nombres pour créer la carte
    x_labels = list(set([int(coord[1]) for coord in coordinates]))
    y_labels = list(set([float(coord[0]) for coord in coordinates]))
    x_labels.sort()
    y_labels.sort()
    x_indices = [x_labels.index(int(coord[1])) for coord in coordinates]
    y_indices = [y_labels.index(float(coord[0])) for coord in coordinates]

    # Créer la matrice pour la carte de fréquentation
    heatmap = np.zeros((len(y_labels), len(x_labels)))
    for i, value in enumerate(values):
        heatmap[y_indices[i], x_indices[i]] = value

    # Afficher la carte de fréquentation avec la colormap
    fig = plt.figure()
    plt.imshow(heatmap, cmap="Blues", interpolation="nearest", vmin=0, vmax=10)
    plt.xticks(range(len(x_labels)), x_labels)
    plt.yticks(range(len(y_labels)), [f"{y:.0e}" for y in y_labels])
    plt.ylabel("Learning Rate")
    plt.xlabel("Batch Size")
    plt.colorbar(label="Experiences", pad=0.02)

    # Ajouter les valeurs à l'intérieur des cellules
    for j, batch in enumerate(x_labels):
        for i, lr in enumerate(y_labels):
            value  = heatmap[i, j]
            if value < 6: color = "black"
            else: color = "white"
            if (str(lr), str(batch)) in grid.keys() \
            or (f"{lr:.0e}", str(batch)) in grid.keys():
                is_running = active[(str(lr), str(batch))]
                if is_running:
                    plt.scatter(j + 0.36, i - 0.36, marker="o", color="orange", s=100)
                plt.text(j, i, str(int(value)), color=color, fontweight="bold", ha="center", va="center")

    # Tracer les lignes horizontales pour délimiter les cellules
    for i in range(len(y_labels)): plt.axhline(i + 0.5, color="black")

    # Tracer les lignes verticales pour délimiter les cellules
    for j in range(len(x_labels)): plt.axvline(j + 0.5, color="black")


    return fig

if __name__ == "__main__":
    matplotlib.use("QtAgg")
    drive = Gdrive.identification()
    main(drive, model="ResNet18_SGD_UFN4")
    plt.show()
