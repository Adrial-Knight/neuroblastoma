import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from . import pydrive_wrap as Gdrive
except ImportError:
    import pydrive_wrap as Gdrive

SKIP_FOLDER = ["__Summary__"]

def main(drive, model):
    grid = make_grid(drive, model)
    display_grid(grid, model)

def make_grid(drive, model):
    root_path = f"Stage_Bilbao_Neuroblastoma/G_Collab/backup/{model}"
    root_id   = Gdrive.get_id_from_path(drive, root_path)
    title_list, id_list = Gdrive.list_from_id(drive, root_id)
    grid = {}
    for title, id in tqdm(zip(title_list, id_list), total=len(id_list)):
        if title in SKIP_FOLDER: continue
        exp_list, _ = Gdrive.list_from_id(drive, id)
        lr, batch = title.split("_")
        value = len(exp_list)
        if "tmp" in exp_list: value -= 1
        grid[(lr, batch)] = value
    return grid

def display_grid(grid, model):
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
    plt.yticks(range(len(y_labels)), y_labels)
    plt.ylabel("Learning Rate")
    plt.xlabel("Batch Size")
    plt.colorbar(label="Experiences", pad=0.02)

    # Ajouter les valeurs à l'intérieur des cellules
    for j, batch in enumerate(x_labels):
        for i, lr in enumerate(y_labels):
            value = heatmap[i, j]
            if value < 6: color = "black"
            elif value < 11: color = "white"
            else: color = "red"
            if not (str(lr), str(batch)) in grid.keys():
                color = "red"
            plt.text(j, i, str(int(value)), color=color, fontweight="bold", ha="center", va="center")
    return fig

if __name__ == "__main__":
    drive = Gdrive.identification()
    main(drive, model="Inception3_SGD_CNL2")
    plt.show()
