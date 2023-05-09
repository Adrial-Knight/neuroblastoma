import json
import numpy as np
import matplotlib.pyplot as plt
import tqdm

try:
    from . import goolgeapiclient_wrap as Gdrive
except ImportError:
    import goolgeapiclient_wrap as Gdrive

SUMMARY_FOLDER = "__Summary__"
TMP_FOLDER = "tmp"
DATA_JSON = "data.json"

def summarize_tab(drive, tab_path):
    update = True
    if update:
        _, id = update_json_tab(drive, tab_path)
    else:
        id = Gdrive.get_id_from_path(drive, f"{tab_path}/{SUMMARY_FOLDER}/{DATA_JSON}")
    data = Gdrive.load_json_from_id(drive, id)
    data = Gdrive.from_json_compatible(data)
    print_best_metric(data)
    for metric in ["loss", "accu"]:
        fig = merge_cell_metric(data, metric)
        folder_id = Gdrive.get_id_from_path(drive, f"{tab_path}/{SUMMARY_FOLDER}")
        Gdrive.upload_fig(drive, folder_id, fig, metric)

def summarize_tab_local():
    with open('/home/pierre/Downloads/data.json', 'r') as file:
        data = json.load(file)
    for metric in ["loss", "accu"]:
        merge_cell_metric(data, metric)
    plt.show()

def summarize_cell(cell_path):
    drive = Gdrive.identification()
    tab_path = "/".join(cell_path.split("/")[:-1])
    folder_id = Gdrive.get_id_from_path(drive, f"{tab_path}/{SUMMARY_FOLDER}")
    cell_id = Gdrive.get_id_from_path(drive, cell_path)
    cell = load_cell_attempts(drive, cell_id)
    for metric in ["accu", "loss"]:
        fig, ax = plt.subplots(1, 1)
        for dataset in ["train", "valid"]:
            key   = f"{dataset}_{metric}"
            mean  = np.nanmean(cell[key], axis=0)
            sigma = np.sqrt(np.nanvar(cell[key], axis=0))

            N = len(mean)
            x = np.arange(N)

            ax.plot(mean, linestyle='-', label=dataset)
            ax.fill_between(x, mean + sigma, mean - sigma, alpha=0.1, label='_nolegend_')
            ax.grid(True, color='gray', linewidth=0.5)
            if metric == "accu":
                ax.set_ylim([0.5, 1])
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylim([0, 1])
                ax.set_ylabel("Binary Cross Entropy")
            ax.set_xlim([0, N])

        ax.plot(cell[f"best_epoch_{metric}"], cell[f"best_{metric}"],
                marker=".", linestyle="None", markersize=3, color='red')
        ax.legend(["train", "valid", "best"])
        ax.set_xlabel("Epochs")
        Gdrive.upload_fig(drive, folder_id, fig, cell_path.split("/")[-1] + f"_{metric}")
        ax.set_title(" ".join(cell_path.split("/")[-2].split("_")) + " (" +
                     ", ".join(cell_path.split("/")[-1].split("_")) + ")")
    plt.show()


def update_json_tab(drive, tab_path: str):
    tab_id = Gdrive.get_id_from_path(drive, tab_path)
    cell_names, cell_ids = Gdrive.list_from_id(drive, tab_id)
    index_of_summary = cell_names.index(SUMMARY_FOLDER)
    cell_names.pop(index_of_summary)
    summary_folder_id = cell_ids.pop(index_of_summary)

    tab = {}
    for cell, cell_id in tqdm.tqdm(zip(cell_names, cell_ids), total=len(cell_names)):
        if details:= load_cell_attempts(drive, cell_id):
            tab[cell] = details
    id = Gdrive.save_dic_to_drive(drive, tab, DATA_JSON, summary_folder_id)
    return tab, id

def print_best_metric(data):
    best_accu = 0
    best_loss = 100
    for key, val in data.items():
        best_accu = max(best_accu, np.max(val["best_accu"]))
        best_loss = min(best_loss, np.min(val["best_loss"]))
    print(f"Best accu: {best_accu}")
    print(f"Best loss: {best_loss}")

def find_best_loss(data1, data2):
    merge = np.maximum(data1, data2)
    epoch = np.argmin(merge)
    loss  = max(data1[epoch], data2[epoch])
    return epoch, loss

def find_best_accu(data1, data2):
    merge = np.minimum(data1, data2)
    epoch = np.argmax(merge)
    accu  = min(data1[epoch], data2[epoch])
    return epoch, accu

def load_cell_attempts(drive, cell_id):
    samples_title, samples_id = Gdrive.list_from_id(drive, cell_id)
    if TMP_FOLDER in samples_title:
        idx_tmp = samples_title.index(TMP_FOLDER)
        samples_title.pop(idx_tmp)
        samples_id.pop(idx_tmp)
    N = len(samples_title)
    if N == 0:
        return None
    # ordering samples
    samples_title, samples_id = zip(*sorted(zip(samples_title, samples_id)))
    details_list = []
    E = 0  # maximum of epoch in a tab-cell
    for sample, folder_id in zip(samples_title, samples_id):
        id = Gdrive.get_id_from_folder_id(drive, folder_id, "details.json")
        details = Gdrive.load_json_from_id(drive, id)
        details_list.append(details)
        if E < details["learning"]["epoch"]:
            E = details["learning"]["epoch"]
    E += 1  # epoch 0
    cell = {"best_epoch_loss": np.zeros(N), "best_loss": np.zeros(N),
            "best_epoch_accu": np.zeros(N), "best_accu": np.zeros(N),
            "train_loss": np.zeros((N, E)), "train_accu": np.zeros((N, E)),
            "valid_loss": np.zeros((N, E)), "valid_accu": np.zeros((N, E))}
    for i, details in enumerate(details_list):
        epoch_l, loss = find_best_loss(details["train_loss"], details["valid_loss"])
        epoch_a, accu = find_best_accu(details["train_accu"], details["valid_accu"])
        cell["best_epoch_loss"][i] = epoch_l
        cell["best_epoch_accu"][i] = epoch_a
        cell["best_loss"][i] = loss
        cell["best_accu"][i] = accu
        for line in ["train_loss", "train_accu", "valid_loss", "valid_accu"]:
            cell[line][i, :] = np.pad(details[line], (0, E-len(details[line])), constant_values=np.nan)
    return cell

def map_cells_to_coord(data):
    cells = list(data.keys())
    lr = sorted(list(set([float(cell.split('_')[0]) for cell in cells])))
    batch = sorted(list(set([int(cell.split('_')[1]) for cell in cells])))
    cell_map = {}
    for i, l in enumerate(lr):
        for j, b in enumerate(batch):
            cell_map[f"{l}_{b}"] = (i, j)
    return cell_map, (lr, batch)

def merge_cell_metric(data, metric):
    fontsize = 14
    if metric not in ["accu", "loss"]:
        raise ValueError("La variable 'metric' ne peut prendre que les valeurs 'accu' ou 'loss'.")
    cell_map, (lr, batch) = map_cells_to_coord(data)
    h, w = max(2, len(lr)), max(2, len(batch))
    fig, axs = plt.subplots(h, w, figsize=(16, 16))
    plt.rcParams.update({'font.size': fontsize})
    for cell_id, cell in data.items():
        coord = cell_map[cell_id]
        for dataset in ["train", "valid"]:
            key   = f"{dataset}_{metric}"
            mean  = np.nanmean(cell[key], axis=0)
            sigma = np.sqrt(np.nanvar(cell[key], axis=0))

            N = len(mean)
            x = np.arange(N)

            axs[coord].plot(mean, linestyle='-', label=dataset)
            axs[coord].fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
            axs[coord].grid(True, color='gray', linewidth=0.5)
            if metric == "accu":
                axs[coord].set_ylim([0.5, 1])
            else:
                axs[coord].set_ylim([0, 1])
            axs[coord].set_xlim([0, N])

    for cell_id, cell in data.items():
        coord = cell_map[cell_id]
        axs[coord].plot(cell[f"best_epoch_{metric}"], cell[f"best_{metric}"],
                        marker=".", linestyle="None", markersize=3, color='red')

    for j, b in enumerate(batch):
        axs[(0, j)].set_title(f"batch={int(b)}")
    for i, l in enumerate(lr):
        axs[(i, 0)].set_ylabel(f"lr={float(l)}", fontsize=fontsize+2, fontstyle='normal', rotation=0)
        axs[(i, 0)].yaxis.set_label_coords(-0.3, 0.45)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    path = "Stage_Bilbao_Neuroblastoma/G_Collab/backup"
    backbones = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "VGG11", "VGG13", "VGG16", "VGG19"]
    backbones = ["Inception3_SGD_CNL2"]
    drive = Gdrive.identification()
    for b in backbones:
        print(b)
        summarize_tab(drive, f"{path}/{b}")
        print("", end="\n")
