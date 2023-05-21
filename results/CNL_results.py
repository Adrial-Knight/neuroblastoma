import matplotlib.pyplot as plt
import tqdm
import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import drive.goolgeapiclient_wrap as Gdrive
import drive.custom_sort as CustomSort


def get_best_metrics(root_path, backbones, tag, exact_end):
    drive = Gdrive.identification()
    root_id = Gdrive.get_id_from_path(drive, root_path)

    idx_list = []
    # Ajout des dossiers cibles commençant par backbones suivi du tag
    network_name_list, network_id_list = Gdrive.list_from_id(drive, root_id)
    backbone_tags  = [f"{b}{tag}" for b in backbones]
    for network in network_name_list:
        for bt in backbone_tags:
            if network.startswith(bt):
                idx = network_name_list.index(network)
                idx_list.append(idx)
    # Ajout des dossiers cibles commençant par backbones et se terminant par exact_end
    for b in backbones:
        idx = network_name_list.index(f"{b}{exact_end}")
        idx_list.append(idx)

    # Filtrage et trie alphanumérique
    network_name_list = [network_name_list[idx] for idx in idx_list]
    network_id_list   = [network_id_list[idx]   for idx in idx_list]
    network_list = sorted(zip(network_name_list, network_id_list),
                        key=lambda x: CustomSort.alphanum(x[0]))

    # Récupération de la meilleur loss
    loss_dico = {}
    for network in tqdm.tqdm(network_list):
        network_name, network_id = network
        summary_id = Gdrive.get_id_from_folder_id(drive, network_id, "__Summary__")
        data_id = Gdrive.get_id_from_folder_id(drive, summary_id, "data.json")
        data = Gdrive.load_json_from_id(drive, data_id)
        loss_list = []
        for key, val in data.items():
            loss_list += val["best_loss"]
        loss = min(loss_list)
        backbone = network_name.split("_")[0]
        if backbone in loss_dico.keys():
            loss_dico[backbone].append(loss)
        else:
            loss_dico[backbone] = [loss]

    return loss_dico


def display(loss_dico):
    plt.figure()

    # Zone de loss sans entraînement
    random_max = 0.7056 + 0.0444
    random_min = 0.7056 - 0.0444
    N = max(len(val) for val in loss_dico.values())
    plt.fill_between(range(N), random_min, random_max, color="red", alpha=0.3, linestyle="dashed", edgecolor="black")
    text_x = N // 2
    text_y = (random_max + random_min) / 2
    plt.text(text_x, text_y, "Loss without training", ha="center", va="center", color="black", fontsize=11)

    # Loss sur les backbones
    colors = ["C0", "C1", "C3", "C2"]
    for c, (key, val) in zip(colors, loss_dico.items()):
        plt.plot(val, "-s", color=c, label=key)

    # Paramètres figure
    plt.ylabel("Loss")
    plt.xlabel("Number of non-linearities added")
    plt.legend(title="Backbone", loc="center right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    root_path = "Stage_Bilbao_Neuroblastoma/G_Collab/backup"
    backbones = ["Inception3", "ResNet18", "ResNet152", "VGG19"]
    pattern = "_SGD_CNL"
    exact_end = "_SGD"
    loss_dico = get_best_metrics(root_path, backbones, pattern, exact_end)
    display(loss_dico)