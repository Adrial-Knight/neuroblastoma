import goolgeapiclient_wrap as Gdrive
import json
import io
import numpy as np
import matplotlib.pyplot as plt


def main():
    drive = Gdrive.identification()
    root = "Stage_Bilbao_Neuroblastoma/G_Collab/backup"
    folder_to_skip = ["LeakTest", "__Summary__", "tmp", "__Results__"]
    root_id = Gdrive.get_id_from_path(drive, root)
    backbones, backbones_id = Gdrive.list_from_id(drive, root_id)

    for backbone, backbone_id in zip(backbones, backbones_id):
        if backbone in folder_to_skip: continue
        cells, cells_id = Gdrive.list_from_id(drive, backbone_id)
        for cell, cell_id in zip(cells, cells_id):
            if cell in folder_to_skip: continue
            exps, exps_id = Gdrive.list_from_id(drive, cell_id)
            for exp, exp_id in zip(exps, exps_id):
                if exp in folder_to_skip: continue
                files, files_id = Gdrive.list_from_id(drive, exp_id)
                for file, file_id in zip(files, files_id):
                    if file.endswith(".pth"):
                        score = Gdrive.load_json_from_id(drive, files_id[files.index("details.json")])["best"]
                        metric = file.split("_")[1].split(".")[0]
                        epoch = score[f"epoch_{metric}"]
                        value = score[metric]
                        owner = Gdrive.get_owner_from_file_id(drive, file_id)
                        if metric == "accu":
                            print(f"Accu={value:.2%}", end=" ")
                        else:
                            print(f"Loss={round(value, 4)}", end=" ")
                        print(f"(epoch {epoch}) in {backbone}/{cell}/{exp} (owner: {owner})")

if __name__ == "__main__":
    main()
