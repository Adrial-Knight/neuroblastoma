import pydrive_wrap as Gdrive
import json
import numpy as np


def main():
    drive = Gdrive.identification()
    tab_path = "Stage_Bilbao_Neuroblastoma/G_Collab/backup/ResNet18_SGD"
    tab_id = Gdrive.get_id_from_path(drive, tab_path)
    cell_names, cell_ids = Gdrive.list_from_id(drive, tab_id)
    index_of_summary = cell_names.index("__Summary__")
    cell_names.pop(index_of_summary)
    summary_folder_id = cell_ids.pop(index_of_summary)

    tab = {}
    for cell in cell_names:
        cell_path = f"{tab_path}/{cell}"
        details = load_cell_attempts(drive, cell_path)
        tab[cell] = details

    Gdrive.save_dic_to_drive(drive, tab, "data.json", summary_folder_id)


def load_cell_attempts(drive, cell_path, E=101):
    cell_id = Gdrive.get_id_from_path(drive, cell_path)
    samples_title, samples_id = Gdrive.list_from_id(drive, cell_id)
    N = len(samples_title)
    cell = {"best_epoch_loss": np.zeros(N), "best_loss": np.zeros(N),
            "best_epoch_accu": np.zeros(N), "best_accu": np.zeros(N),
            "train_loss": np.zeros((N, E)), "train_accu": np.zeros((N, E)),
            "valid_loss": np.zeros((N, E)), "valid_accu": np.zeros((N, E))}
    for i, sample in enumerate(samples_title):
        id = Gdrive.get_id_from_path(drive, f"{cell_path}/{sample}/details.json")
        details = Gdrive.load_json_from_id(drive, id)
        cell["best_epoch_loss"][i] = details["best"]["epoch_loss"]
        cell["best_epoch_accu"][i] = details["best"]["epoch_accu"]
        cell["best_loss"][i] = details["best"]["loss"]
        cell["best_accu"][i] = details["best"]["accu"]
        for line in ["train_loss", "train_accu", "valid_loss", "valid_accu"]:
            cell[line][i, :] = np.pad(details[line], (0, E-len(details[line])), constant_values=np.nan)
    return cell


if __name__ == "__main__":
    main()
