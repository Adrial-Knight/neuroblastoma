import matplotlib.pyplot as plt
import numpy as np
import re
import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import drive.goolgeapiclient_wrap as Gdrive
from drive import custom_sort

SUMMARY_FOLDER = "__Summary__"
RESULTS_FOLDER = "__Results__"
SKIP_FOLDER = ["LeakTest", RESULTS_FOLDER]

def main(path, pattern):
    drive = Gdrive.identification()
    data, folder_id = extract_result(drive, path)
    names = filter_list_by_pattern(data, pattern)
    accu  = [data[name]["accu"]["best"] for name in names]
    loss  = [data[name]["loss"]["best"] for name in names]
    names = [name.split("_")[0] for name in names]
    # names = clean_names(names, pattern)
    print(names)
    fig = plot_accu_loss(names, accu, loss)
    Gdrive.upload_fig(drive, folder_id, fig, pattern.strip("*").strip("_"))
    plt.show()

def clean_names(names, pattern):
    names_ = []
    for name in names:
        match = re.match(pattern + r'(_\w+)?', name)
        if match:
            names_.append(match.group().split("_SGD")[0])
    return names_


def filter_list_by_pattern(lst, pattern):
    if "*" not in pattern:
        raise ValueError("Pattern must contain a star (*).")

    star_index = pattern.index("*")

    if star_index == 0:
        filtered_list = list(filter(lambda x: x.endswith(pattern[1:]), lst))
    elif star_index == len(pattern) - 1:
        filtered_list = list(filter(lambda x: x.startswith(pattern[:-1]), lst))
    else:
        raise ValueError("The star (*) must be at the beginnig or at the end.")

    filtered_list = sorted(filtered_list, key=custom_sort.alphanum)

    return filtered_list

def extract_result(drive, backup_path):
    backup_id = Gdrive.get_id_from_path(drive, backup_path)
    network_titles, network_ids = Gdrive.list_from_id(drive, backup_id)

    def extract_result_network(network_id):
        sum_folder_id = Gdrive.get_id_from_folder_id(drive, network_id, SUMMARY_FOLDER)
        data_id = Gdrive.get_id_from_folder_id(drive, sum_folder_id, "data.json")
        data = Gdrive.load_json_from_id(drive, data_id)
        Gdrive.to_json_compatible(data)

        result = {"loss":
                    {"best": 1000, "epoch": -1, "lr": -1, "batch": -1, "exp": -1},
                  "accu":
                    {"best": 0.00, "epoch": -1, "lr": -1, "batch": -1, "exp": -1}}
        for key, val in data.items():
            exp_l, loss = min(enumerate(val["best_loss"]), key=lambda x: x[1])
            exp_a, accu = max(enumerate(val["best_accu"]), key=lambda x: x[1])
            lr, batch = key.split("_")
            lr = float(lr)
            batch = int(batch)
            if loss < result["loss"]["best"]:
                result["loss"]["best"]  = loss
                result["loss"]["epoch"] = val["best_epoch_loss"][exp_l]
                result["loss"]["lr"] = lr
                result["loss"]["batch"] = batch
                result["loss"]["exp"] = exp_l + 1
            if accu > result["accu"]["best"]:
                result["accu"]["best"]  = accu
                result["accu"]["epoch"] = val["best_epoch_accu"][exp_a]
                result["accu"]["lr"] = lr
                result["accu"]["batch"] = batch
                result["accu"]["exp"] = exp_a + 1
        return result

    data = {}
    for title, id in zip(network_titles, network_ids):
        if title in SKIP_FOLDER: continue
        try:
            result = extract_result_network(id)
        except IndexError:
            continue
        data[title] = result
        print(f"{title}")
        print(f"\tLoss={result['loss']['best']:.3f}  (epoch: {int(result['loss']['epoch']):2d}) | (lr, batch)=({result['loss']['lr']}, {result['loss']['batch']:3d}) at exp {result['loss']['exp']}")
        print(f"\tAccu={result['accu']['best']:.2%} (epoch: {int(result['accu']['epoch']):2d}) | (lr, batch)=({result['accu']['lr']}, {result['accu']['batch']:3d}) at exp {result['accu']['exp']}")
    results_id = network_ids[network_titles.index(RESULTS_FOLDER)]
    Gdrive.save_dic_to_drive(drive, data, "result.json", results_id)
    return data, results_id

def plot_accu_loss(names, accuracy, loss):
    fontsize = 14
    if max(accuracy) <= 1:
        accuracy = [round(a * 100, 2) for a in accuracy]

    # Template
    bar_width = 0.4
    fig, ax1 = plt.subplots(num="Accuracy and Loss Results", figsize=(15, 6))
    ax1.bar(names, accuracy, color='#1f77b4', width=bar_width)
    ax1.set_xticks(np.arange(len(names)) + bar_width/2)
    ax1.tick_params(axis='both', labelsize=fontsize)
    ax2 = ax1.twinx()
    ax2.bar([i+bar_width for i in range(len(names))], loss, color='#ff7f0e', width=bar_width)
    ax2.tick_params(axis='both', labelsize=fontsize)

    # Add values
    for i, (a, l) in enumerate(zip(accuracy, loss)):
        ax1.text(i, a + 0.5, f"{a:.1f}%", color='black', ha='center', fontsize=fontsize)
        ax2.text(i+bar_width, l+0.005, f"{l:.3f}", color='black', ha='center', fontsize=fontsize)

    # Add red borders between different network families (eg: ResNet | VGG)
    for i in range(1, len(names)):
        if names[i - 1][:3] != names[i][:3]:
            x_center = (2*i - 1 + bar_width) / 2
            ax1.axvline(x=x_center, color='r', linestyle='--', alpha=0.7)

    # Axe limits
    ax1.set_ylim([0, 100])
    ax2.set_ylim([0, 1])

    # Labels
    ax1.set_xlabel("Backbone", fontsize=fontsize)
    ax1.set_ylabel("Accuracy (%)", fontsize=fontsize)
    ax2.set_ylabel("Loss", fontsize=fontsize)

    # Modern look
    ax1.set_facecolor('#f0f0f0')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    path = "Stage_Bilbao_Neuroblastoma/G_Collab/backup"
    # pattern = "Inception3*"
    pattern = "*_SGD"
    main(path, pattern)
