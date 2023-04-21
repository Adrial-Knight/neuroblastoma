import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import shutil
from math import floor, ceil


FIG_FOLDER = "figures"

def get_families(path):
    fnames = [fname.split(".")[0] for fname in os.listdir(path)]
    patchs = [fname.rsplit("_", 1) for fname in fnames]
    label  = path.rsplit("/", 1)[-1]
    family_infos = {"path": path, "label": label, "files": {}}
    for element in patchs:
        key = element[0]
        value = int(element[1])
        if key in family_infos["files"]:
            family_infos["files"][key].append(value)
        else:
            family_infos["files"][key] = [value]
    return family_infos

def compute_proportions(family_infos):
    path = family_infos["path"]
    for fam_name, patches in family_infos["files"].items():
        family_infos[fam_name] = 0
        for patch_id in patches:
            with Image.open(f"{path}/{fam_name}_{patch_id}.jpg") as img:
                W, H = img.size
                family_infos[fam_name] += W * H

    categories = list(family_infos["files"].keys())
    total = sum([family_infos[key] for key in categories])
    proportions = [family_infos[key]/total for key in categories]

    d = {"category":categories, "proportions":proportions}
    sorted_data = sorted(zip(d["category"], d["proportions"]), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_proportions = zip(*sorted_data)

    family_infos["category"] = list(sorted_labels)
    family_infos["proportions"] = list(sorted_proportions)

def split_PSP(family_infos, p=[.7, .15, .15]):
    categories  = family_infos["category"].copy()
    proportions = family_infos["proportions"].copy()
    parts = [{"category": [], "capacity": c} for c in p]

    def affect(proportions, categories, P):
        original_capacity = P["capacity"]
        while P["capacity"] > 0 and categories:
            P["category"].append(categories.pop(0))
            P["capacity"] -= proportions.pop(0)
        P["capacity"] = original_capacity - P["capacity"]

    for P in parts:
        affect(proportions, categories, P)

    return parts

def split_IFP(family_infos, p=[.7, .15, .15], mu=1):
    return split_IFPO(family_infos, p, mu, extra_capacity=0)

def split_IFPO(family_infos, p=[.7, .15, .15], mu=1, extra_capacity=0):
    categories = family_infos["category"].copy()
    proportions = family_infos["proportions"].copy()
    parts = [{"category": [], "capacity": c} for c in p]
    total_categories = len(categories)
    threshold = mu*min(parts[1]["capacity"], parts[2]["capacity"])
    while categories and proportions[0] > threshold:
        parts[0]["category"].append(categories.pop(0))
        parts[0]["capacity"] -= proportions.pop(0)

    def affect(proportions, categories, P):
        if not categories: return False
        for i, prop in enumerate(proportions):
            if P["capacity"] > prop:
                P["capacity"] -= proportions.pop(i)
                P["category"].append(categories.pop(i))
                return True
        return False

    deficits = [0, len(parts[0]["category"]), len(parts[0]["category"])]
    while sum(deficits) > 0:
        for i, P in enumerate(parts):
            if deficits[i]:
                if affect(proportions, categories, P):
                    deficits[i] -= 1
                else:
                    deficits[i] = 0

    for P in parts[1:]: P["capacity"] += extra_capacity
    pending = [True, True, True]
    while categories and any(pending):
        for i, P in enumerate(parts):
            pending[i] = affect(proportions, categories, P)
    for P in parts[1:]: P["capacity"] -= extra_capacity

    if categories:
        parts[0]["category"] += categories
        parts[0]["capacity"] -= sum(proportions)

    for c, P in zip(p, parts): P["capacity"] = c - P["capacity"]

    capacity  = np.array([parts[i]["capacity"] for i in [0, 1, 2]])
    diversity = np.array([len(parts[i]["category"])/total_categories for i in [0, 1, 2]])

    return parts, capacity, diversity

def optimize_IFP(family_infos, prop, mu_list):
    loss_capacity  = np.zeros((len(mu_list)))
    loss_diversity = np.zeros((len(mu_list)))
    for i_mu, mu in enumerate(mu_list):
        parts, capacity, diversity = split_IFP(family_infos, prop, mu)
        loss_capacity[i_mu]  = np.mean(np.abs(capacity - prop))
        loss_diversity[i_mu] = np.mean(np.abs(diversity - 1/3))
    i_min = np.argmin(loss_capacity + loss_diversity)
    mu_opt = mu_list[i_min]
    parts, _, _ = split_IFP(family_infos, prop, mu_opt)
    metrics = {"capacity": loss_capacity,
               "diversity": loss_diversity,
               "mu": mu_opt}
    return parts, metrics

def optimize_IFPO(family_infos, prop, mu_list, eps_list):
    heatmap_capacity  = np.zeros((len(mu_list), len(eps_list)))
    heatmap_diversity = np.zeros((len(mu_list), len(eps_list)))
    for i_mu, mu in enumerate(mu_list):
        for i_eps, eps in enumerate(eps_list):
            _, capacity, diversity = split_IFPO(family_infos, prop, mu, eps)
            heatmap_capacity[i_mu, i_eps]  = np.mean(np.abs(capacity - prop))
            heatmap_diversity[i_mu, i_eps] = np.mean(np.abs(diversity - 1/3))
    min_idx = np.argmin(heatmap_capacity + heatmap_diversity)
    min_idx_2d = np.unravel_index(min_idx, heatmap_capacity.shape)
    mu_opt, eps_opt = mu_list[min_idx_2d[0]], eps_list[min_idx_2d[1]]
    parts, _, _ = split_IFPO(family_infos, prop, mu_opt, eps_opt)
    metrics = {"capacity": heatmap_capacity,
               "diversity": heatmap_diversity,
               "mu": mu_opt,
               "eps": eps_opt}
    return parts, metrics

def print_parts(parts, method="", mu=None, eps=None):
    capacity  = tuple([int(P["capacity"]*10000)/100 for P in parts])
    diversity = tuple([len(P["category"]) for P in parts])
    print(f"\t{method}:")
    print(f"\t{capacity= }")
    print(f"\t{diversity=}")
    if mu: mu=round(mu, 2)
    if eps: eps=round(eps, 2)
    print(f"\t{(mu, eps)=}\n")

def savefig_family_repartition(family_infos):
    fig, ax = plt.subplots(num=f"{family_infos['label']} Class Balance", nrows=1, ncols=1)
    labels  = [f"f{i}" for i in range(1, len(family_infos["category"])+1)]
    ax.pie(family_infos["proportions"], labels=labels,
           autopct="%1.1f%%", textprops={"fontsize": 9})
    for i, prop in enumerate(family_infos["proportions"]):
        if prop < 0.05:
            plt.setp(ax.texts[2*i],   visible=False)
            plt.setp(ax.texts[2*i+1], visible=False)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)
    fig.set_size_inches(2.5, 2.5)
    plt.savefig(f"{FIG_FOLDER}/class_balance_{family_infos['label']}.pdf",
                format="pdf", facecolor=None, pad_inches=0, bbox_inches="tight",
                dpi=1200)
    return fig

def savefig_partition(family_infos, parts, method):
    fig, ax = plt.subplots(num=f"{method} Partitioning - {family_infos['label']} Class", nrows=1, ncols=1)
    proportions = [P["capacity"] if P["capacity"] > 0 else 0 for P in parts]
    datasets = ["P1", "P2", "P3"]
    labels = [f"{dataset} ({len(P['category'])})" for dataset, P in zip(datasets, parts)]
    ax.pie(proportions, labels=labels, autopct="%1.1f%%", textprops={"fontsize": 11})
    ax.axis("equal")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.margins(0, 0)
    fig.set_size_inches(2.5, 2.5)
    plt.savefig(f"{FIG_FOLDER}/partition_{method}_{family_infos['label']}.pdf",
                format="pdf", facecolor=None, pad_inches=0, bbox_inches="tight",
                dpi=1200)
    return fig

def savefig_loss(family_infos, loss_metrics, mu_list):
    fig, ax = plt.subplots(num=f"IFP Loss - {family_infos['label']} Class", nrows=1, ncols=1, figsize=(9, 4))
    loss_capacity = loss_metrics["capacity"]
    loss_diversity = loss_metrics["diversity"]
    mean_loss = (loss_capacity + loss_diversity) / 2
    ax.plot(mu_list, loss_capacity, label="Loss Capacity")
    ax.plot(mu_list, loss_diversity, label="Loss Diversity")
    ax.plot(mu_list, mean_loss, linestyle="--", label="Loss Mean")
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("Loss")
    ax.grid()
    ax.legend()
    plt.savefig(f"{FIG_FOLDER}/loss_IFP_{family_infos['label']}.pdf", format="pdf", bbox_inches="tight", pad_inches=0)
    return fig

def savefig_heatmap(family_infos, metrics, mu_list, eps_list, type):
    heatmap = metrics[type]
    fig, ax = plt.subplots(num=f"Heatmap {type} - {family_infos['label']} Class")
    im = plt.imshow(heatmap, cmap="coolwarm", origin="lower",
                    extent=[eps_list[0], eps_list[-1], mu_list[0], mu_list[-1]])
    ax.set_xlabel("Extra Capacity $\epsilon$")
    ax.set_ylabel("$\mu$")
    ax.set_aspect("auto")
    fig.colorbar(im)
    plt.savefig(f"{FIG_FOLDER}/heatmap_{type}_{family_infos['label']}.pdf",
                format="pdf", bbox_inches="tight", pad_inches=0)
    return fig

def split_files(db, prop=[.7, .15, .15], class_names=["PD", "D"]):
    class_infos = {}
    for cname in class_names:
        family_infos = get_families(path=f"{db}/{cname}")
        compute_proportions(family_infos)
        class_infos[cname] = family_infos

    mu_list = np.linspace(0, 1, 1000)
    eps_list = np.linspace(0, 0.1, 100)
    partitions = {}
    for cname, family_infos in class_infos.items():
        parts, _ = optimize_IFPO(family_infos, prop, mu_list, eps_list)
        partitions[cname] = parts

    datasets = {"train": {"files": {}, "nb_im": {}},
                "valid": {"files": {}, "nb_im": {}},
                "test":  {"files": {}, "nb_im": {}}}
    for dataset_id, dataset in enumerate(datasets.values()):
        for cname, parts in partitions.items():
            patterns = parts[dataset_id]["category"]
            path = f"{db}/{cname}"
            files = [file for file in os.listdir(path) if any(file.startswith(pattern) for pattern in patterns)]
            dataset["files"][cname] = files
            dataset["nb_im"][cname] = len(files)

    with open(f"{db}/split.json", "w") as fd:
        json.dump(datasets, fd, sort_keys=True, indent=4)
    return datasets

def assign_files(old_root: str, datasets: json):
    new_root = old_root + "_split"
    os.makedirs(new_root, exist_ok=True)
    for dataset_id, infos in datasets.items():
        classes = infos["files"]
        dataset_dir = f"{new_root}/{dataset_id}"
        for class_id, files in classes.items():
            old_class_dir = f"{old_root}/{class_id}"
            new_class_dir = f"{dataset_dir}/{class_id}"
            os.makedirs(new_class_dir, exist_ok=True)
            for file in files:
                src_file = os.path.join(old_class_dir, file)
                dst_file = os.path.join(new_class_dir, file)
                shutil.copy(src_file, dst_file)
    with open(f"{new_root}/split.json", "w") as fd:
        json.dump(datasets, fd, sort_keys=True, indent=4)
    return new_root

def geometric_transform(image, nt):
    im_list = [image, np.fliplr(image)]
    id_list = ["original", "flip"]
    for i in range(1, ceil(nt/2) + 1):
        im_list.append(np.rot90(im_list[-2]))
        im_list.append(np.rot90(im_list[-2]))
        id_list.append(str(90*i))
        id_list.append("flip" + str(90*i))
    return im_list[:nt+1], id_list[:nt+1]

def augment_and_balance(root: str, datasets: json):
    t = 7
    new_root = f"{root}_BA"
    os.makedirs(new_root, exist_ok=True)
    for dataset_id, dataset in datasets.items():
        os.makedirs(f"{new_root}/{dataset_id}", exist_ok=True)
        n_im_list = list(dataset["nb_im"].values())
        n_im_max, n_im_min = max(n_im_list), min(n_im_list)
        r = round(n_im_max / n_im_min)
        ngca = floor((t+1)/r)-1 if t >= r else 0
        for class_id, n_im in dataset["nb_im"].items():
            os.makedirs(f"{new_root}/{dataset_id}/{class_id}", exist_ok=True)
            r = round(n_im_max / n_im)
            if r == 1: nt = ngca
            else:
                if t >= r-1: nt = r * ngca + r - 1
                else: nt = t
            class_path = f"{root}/{dataset_id}/{class_id}"
            new_class_path = f"{new_root}/{dataset_id}/{class_id}"
            file_names = dataset["files"][class_id]
            for file_name in file_names:
                file_path = f"{class_path}/{file_name}"
                im = plt.imread(file_path)
                im_list, id_list = geometric_transform(im, nt)
                base_name = file_name.split(".")[0]
                for im, id in zip(im_list, id_list):
                    fpath = f"{new_class_path}/{base_name}_{id}.jpg"
                    Image.fromarray(im).save(fpath, format="JPEG", quality=100)

def print_train_mean_deviation(root: str):
    R_sum, G_sum, B_sum, num_pixels = 0, 0, 0, 0
    labels = os.listdir(os.path.join(root, 'train'))

    for label in labels:
        for img_file in os.listdir(os.path.join(root, f"train/{label}")):
            img_path = os.path.join(root, f"train/{label}", img_file)
            img = np.asarray(Image.open(img_path))

            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]

            R_sum += np.sum(R, dtype=np.uint64)
            G_sum += np.sum(G, dtype=np.uint64)
            B_sum += np.sum(B, dtype=np.uint64)

            num_pixels += img.shape[0] * img.shape[1]

    R_mean = R_sum / num_pixels
    G_mean = G_sum / num_pixels
    B_mean = B_sum / num_pixels

    # Vérifier s'il y a eu un overflow dans le calcul de la somme des pixels
    if R_sum > np.iinfo(np.uint64).max or G_sum > np.iinfo(np.uint64).max or B_sum > np.iinfo(np.uint64).max:
        print("Overflow détecté lors du calcul de la somme des pixels.")

    R_var_sum, G_var_sum, B_var_sum = 0, 0, 0
    for label in labels:
        for img_file in os.listdir(os.path.join(root, f"train/{label}")):
            img_path = os.path.join(root, f"train/{label}", img_file)
            img = np.asarray(Image.open(img_path))

            R = img[:,:,0]
            G = img[:,:,1]
            B = img[:,:,2]

            R_var_sum += np.sum((R - R_mean) ** 2, dtype=np.float64)
            G_var_sum += np.sum((G - G_mean) ** 2, dtype=np.float64)
            B_var_sum += np.sum((B - B_mean) ** 2, dtype=np.float64)

    R_std = np.sqrt(R_var_sum / num_pixels)
    G_std = np.sqrt(G_var_sum / num_pixels)
    B_std = np.sqrt(B_var_sum / num_pixels)

    print(f"{(R_mean/255, G_mean/255, B_mean/255)}")
    print(f"{(R_std/255, G_std/255, B_std/255)}")

def force_augmentation(old_root: str, augmentation: dict):
    new_root = f"{old_root}_FA"
    os.makedirs(new_root, exist_ok=True)
    datasets = os.listdir(old_root)
    for dataset in datasets:
        old_dataset_root = f"{old_root}/{dataset}"
        new_dataset_root = f"{new_root}/{dataset}"
        os.makedirs(new_dataset_root, exist_ok=True)
        labels = os.listdir(old_dataset_root)
        for label in labels:
            if label in augmentation:
                old_class_path = f"{old_dataset_root}/{label}"
                new_class_path = f"{new_dataset_root}/{label}"
                os.makedirs(new_class_path, exist_ok=True)
                file_names = os.listdir(old_class_path)
                for file_name in file_names:
                    img_path = f"{old_class_path}/{file_name}"
                    im = plt.imread(img_path)
                    im_list, id_list = geometric_transform(im, augmentation[label])
                    base_name = file_name.split(".")[0]
                    for im, id in zip(im_list, id_list):
                        fpath = f"{new_class_path}/{base_name}_{id}.jpg"
                        Image.fromarray(im).save(fpath, format="JPEG", quality=100)

def test(db, prop=[.7, .15, .15]):
    class_infos = {}
    for cname in ["PD", "D"]:
        family_infos = get_families(path=f"{db}/{cname}")
        compute_proportions(family_infos)
        class_infos[cname] = family_infos

    os.makedirs(FIG_FOLDER, exist_ok=True)

    for cname in ["D", "PD"]:
        family_infos = class_infos[cname]
        total_families = len(family_infos["files"].keys())
        print(f"{cname} Class: {total_families} families")
        savefig_family_repartition(family_infos)

        # PSP: Proportional and Sequential Partitioning
        parts = split_PSP(family_infos, prop)
        print_parts(parts, method="PSP")
        savefig_partition(family_infos, parts, method="PSP")

        # IFP: Iterative Filling Partitioning (1 HP)
        mu_list = np.linspace(0, 1, 1000)
        parts, metrics = optimize_IFP(family_infos, prop, mu_list)
        print_parts(parts, method="IFP", mu=metrics["mu"])
        savefig_partition(family_infos, parts, method="IFP")
        savefig_loss(family_infos, metrics, mu_list)

        # IFPO: Iterative Filling Partitioning with Overfill (2 HP)
        eps_list = np.linspace(0, 0.1, 100)
        parts, metrics = optimize_IFPO(family_infos, prop, mu_list, eps_list)
        print_parts(parts, method="IFP0", mu=metrics["mu"], eps=metrics["eps"])
        savefig_partition(family_infos, parts, method="IFPO")
        savefig_heatmap(family_infos, metrics, mu_list, eps_list, type="capacity")
        savefig_heatmap(family_infos, metrics, mu_list, eps_list, type="diversity")

    # plt.show()


if __name__ == "__main__":
    # db="database/224x224"
    db="database/250x250"
    proportions = [.75, .125, .125]
    # test(db, prop=proportions)
    datasets = split_files(db, prop=proportions)
    new_db = assign_files(db, datasets)
    augment_and_balance(new_db, datasets)
    print_train_mean_deviation(new_db)
    # print_train_mean_deviation(db)

    # dico_augmentation = {"PD": 3, "D": 7}
    # force_augmentation(db, dico_augmentation)
    # print_train_mean_deviation(f"{db}_FA")
