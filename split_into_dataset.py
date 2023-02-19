import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json


def get_families(path):
    fnames = [fname.split(".")[0] for fname in os.listdir(path)]
    patchs = [fname.rsplit("_", 1) for fname in fnames]
    label  = path.rsplit("/", 1)[-1]
    family_info = {"path": path, "label": label, "files": {}}
    for element in patchs:
        key = element[0]
        value = int(element[1])
        if key in family_info["files"]:
            family_info["files"][key].append(value)
        else:
            family_info["files"][key] = [value]
    return family_info

def compute_proportions(family_info):
    path = family_info["path"]
    for fam_name, patches in family_info["files"].items():
        family_info[fam_name] = 0
        for patch_id in patches:
            with Image.open(f"{path}/{fam_name}_{patch_id}.jpg") as img:
                W, H = img.size
                family_info[fam_name] += W * H

    categories = list(family_info["files"].keys())
    total = sum([family_info[key] for key in categories])
    proportions = [family_info[key]/total for key in categories]

    d = {"category":categories, "proportions":proportions}
    sorted_data = sorted(zip(d["category"], d["proportions"]), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_proportions = zip(*sorted_data)

    family_info["category"] = list(sorted_labels)
    family_info["proportions"] = list(sorted_proportions)


def split_categories(family_info, p=[.7, .15, .15], mu=1, extra_capacity=0):
    categories = family_info["category"].copy()
    proportions = family_info["proportions"].copy()
    parts = [{"category": [], "capacity": c} for c in p]
    total_categories = len(categories)
    for P in parts[1:]: P["capacity"] += extra_capacity
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

def display_balance_partition(family_infos, parts):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    axes[0].pie(family_infos["proportions"], labels=family_infos["category"], autopct="%1.1f%%")
    axes[0].set_title(f"{family_infos['label']} Class Balance")

    for i, prop in enumerate(family_infos["proportions"]):
        if prop < 0.05:
            plt.setp(axes[0].texts[2*i], visible=False)
            plt.setp(axes[0].texts[2*i+1], visible=False)

    proportions = [P["capacity"] if P["capacity"] > 0 else 0 for P in parts]
    datasets = ["Training", "Validation", "Test"]
    labels = [f"{dataset} ({len(P['category'])})" for dataset, P in zip(datasets, parts)]
    axes[1].pie(proportions, labels=labels, autopct="%1.1f%%")
    axes[1].set_title(f"{family_infos['label']} Partitions")

def display_loss(family_infos, loss_capacity, loss_diversity, mu_list):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    mean_loss = (loss_capacity + loss_diversity) / 2
    ax.plot(mu_list, loss_capacity, label="Loss Capacity")
    ax.plot(mu_list, loss_diversity, label="Loss Diversity")
    ax.plot(mu_list, mean_loss, label="Loss Mean")
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("Loss")
    ax.set_title(f"{family_infos['label']} Class - Loss Plot")
    ax.grid()
    ax.legend()

def search1D(family_infos, prop, mu_list, extra_capacity):
    loss_capacity  = np.zeros((len(mu_list)))
    loss_diversity = np.zeros((len(mu_list)))
    for i_mu, mu in enumerate(mu_list):
        parts, capacity, diversity = split_categories(family_infos, prop, mu, extra_capacity)
        loss_capacity[i_mu]  = np.mean(np.abs(capacity - prop))
        loss_diversity[i_mu] = np.mean(np.abs(diversity - 1/3))
    i_min = np.argmin(loss_capacity + loss_diversity)
    parts, _, _ = split_categories(family_infos, prop, mu_list[i_min], extra_capacity)
    return parts, loss_capacity, loss_diversity

def search2D(family_infos, prop, mu_list, extra_capacity_list):
    heatmap_capacity  = np.zeros((len(mu_list), len(extra_capacity_list)))
    heatmap_diversity = np.zeros((len(mu_list), len(extra_capacity_list)))
    for i_mu, mu in enumerate(mu_list):
        for i_xc, xc in enumerate(extra_capacity_list):
            _, capacity, diversity = split_categories(family_infos, prop, mu, xc)
            heatmap_capacity[i_mu, i_xc]  = np.mean(np.abs(capacity - prop))
            heatmap_diversity[i_mu, i_xc] = np.mean(np.abs(diversity - 1/3))
    min_idx = np.argmin(heatmap_capacity + heatmap_diversity)
    min_idx_2d = np.unravel_index(min_idx, heatmap_capacity.shape)
    parts, _, _ = split_categories(family_infos, prop, mu_list[min_idx_2d[0]], extra_capacity_list[min_idx_2d[1]])
    return parts, heatmap_capacity, heatmap_diversity

def display_heatmaps(heatmap_capacity, heatmap_diversity, mu_list, extra_capacity_list):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    im = []
    im.append(axs[0].imshow(heatmap_capacity, cmap="coolwarm", origin="lower",
                       extent=[extra_capacity_list[0], extra_capacity_list[-1], mu_list[0], mu_list[-1]]))
    im.append(axs[1].imshow(heatmap_diversity,cmap="coolwarm", origin="lower",
                       extent=[extra_capacity_list[0], extra_capacity_list[-1], mu_list[0], mu_list[-1]]))
    for i, lab in enumerate(["Capacity", "Diversity"]):
        axs[i].set_xlabel("Extra Capacity")
        axs[i].set_ylabel("$\mu$")
        axs[i].set_title(f"Loss {lab}")
        axs[i].set_aspect("auto")
        fig.colorbar(im[i], ax=axs[i])

def split(db, prop=[.7, .15, .15], class_names=["PD", "D", "ED"]):
    class_infos = {}
    for cname in class_names:
        family_infos = get_families(path=f"{db}/{cname}")
        compute_proportions(family_infos)
        class_infos[cname] = family_infos

    mu_list = np.linspace(0, 1, 1000)
    extra_capacity_list = np.linspace(0, 0.1, 100)
    partitions = {}
    for cname, family_infos in class_infos.items():
        # parts, _, _ = search1D(family_infos, prop, mu_list, extra_capacity=.05)
        parts, _, _ = search2D(family_infos, prop, mu_list, extra_capacity_list)
        partitions[cname] = parts

    datasets = {"training":   {"files": {}, "pixels": {}},
                "validation": {"files": {}, "pixels": {}},
                "test":       {"files": {}, "pixels": {}}}
    for dataset_id, dataset in enumerate(datasets.values()):
        for cname, parts in partitions.items():
            patterns = parts[dataset_id]["category"]
            path = f"{db}/{cname}"
            files = [file for file in os.listdir(path) if any(file.startswith(pattern) for pattern in patterns)]
            dataset["files"][cname] = files
            dataset["pixels"][cname] = 0
            for fname in files:
                with Image.open(f"{path}/{fname}") as img:
                    W, H = img.size
                    dataset["pixels"][cname] += W * H

    with open(f"{db}/split.json", "w") as fd:
        json.dump(datasets, fd, sort_keys=True, indent=4)
    return datasets


def test(db):
    class_infos = {}
    prop = [.7, .15, .15]
    for cname in ["PD", "D", "ED"]:
        family_infos = get_families(path=f"{db}/{cname}")
        compute_proportions(family_infos)
        class_infos[cname] = family_infos

    family_infos = class_infos["ED"]

    # 1D: adjust mu
    mu_list = np.linspace(0, 1, 1000)
    extra_capacity = 0.05
    parts, loss_capacity, loss_diversity = search1D(family_infos, prop, mu_list, extra_capacity)
    display_balance_partition(family_infos, parts)
    display_loss(family_infos, loss_capacity, loss_diversity, mu_list)

    # 2D: adjust mu and extra_capacity
    extra_capacity_list = np.linspace(0, 0.1, 100)
    parts, heatmap_capacity, heatmap_diversity = search2D(family_infos, prop, mu_list, extra_capacity_list)
    display_balance_partition(family_infos, parts)
    display_heatmaps(heatmap_capacity, heatmap_diversity, mu_list, extra_capacity_list)

    plt.show()


if __name__ == "__main__":
    # test(db="database/224x224")
    split(db="database/224x224")
