import pydrive_wrap as Gdrive
import os
from tqdm import tqdm

BACKUP = "../backup"
SKIP = {"network": ["LeakTest", "__Results__"], "cell": ["__Summary__"], "exp": ["tmp"]}


def main(path, extension):
    drive = Gdrive.identification()
    root_id = Gdrive.get_id_from_path(drive, path)
    network_name_list, network_id_list = Gdrive.list_from_id(drive, root_id)
    for network_name, network_id in tqdm(zip(network_name_list, network_id_list)):
        if network_name in SKIP["network"]: continue
        cell_name_list, cell_id_list = Gdrive.list_from_id(drive, network_id)
        for cell_name, cell_id in zip(cell_name_list, cell_id_list):
            if cell_name in SKIP["cell"]: continue
            exp_name_list, exp_id_list = Gdrive.list_from_id(drive, cell_id)
            for exp_name, exp_id in zip(exp_name_list, exp_id_list):
                if exp_name in SKIP["exp"]: continue
                exp_path = f"{BACKUP}/{network_name}/{cell_name}/{exp_name}"
                os.makedirs(exp_path, exist_ok=True)
                file_name_list, file_id_list = Gdrive.list_from_id(drive, exp_id)
                file_name_list, file_id_list = list(zip(*list(filter(lambda x: x[0].endswith(extension), zip(file_name_list, file_id_list)))))
                for file_name, file_id in zip(file_name_list, file_id_list):
                    Gdrive.download_file(drive, file_id, f"{exp_path}/{file_name}")

if __name__ == "__main__":
    path = "Stage_Bilbao_Neuroblastoma/G_Collab/backup"
    extension = ".json"
    main(path, extension)
