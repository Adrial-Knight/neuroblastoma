import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def run_test(im_path):
    image = plt.imread(im_path)
    im_list = geometric_transform(image)
    show_transform(im_list)
    save_im_list(im_path, im_list)

def geometric_transform(image):
    im_list = [image, np.fliplr(image)]
    for i in range(3):
        im_list.append(np.rot90(im_list[-2]))
        im_list.append(np.rot90(im_list[-2]))
    return im_list

def save_im_list(original_path, im_list):
    os.makedirs("results", exist_ok=True)
    fname, _ = os.path.splitext(os.path.basename(original_path))
    fpath = f"results/{fname}"
    for i in range(4):
        Image.fromarray(im_list[2*i]).save(f"{fpath}_nf{str(90*i)}.jpg", format="JPEG", quality=100)
        Image.fromarray(im_list[2*i+1]).save(f"{fpath}_fh{str(90*i)}.jpg", format="JPEG", quality=100)

def show_transform(im_list):
    fig = plt.figure(figsize=(12,12))
    fig.suptitle("7 geometric transforms")
    subfigs = fig.subfigures(nrows=2, ncols=1)
    axs_0 = subfigs[0].subplots(1, 4)
    axs_1 = subfigs[1].subplots(1, 4)
    subfigs[0].suptitle("Not flipped")
    subfigs[1].suptitle("Flipped Horizontally")
    for ax in axs_0: ax.axis("off")
    for ax in axs_1: ax.axis("off")
    rotation_titles = ["Rotated " + str(90*i) + "°" for i in range(4)]
    for i in range(4):
        axs_0[i].imshow(im_list[2*i])
        axs_0[i].set_title(rotation_titles[i])
        axs_1[i].imshow(im_list[2*i+1])
        axs_1[i].set_title(rotation_titles[i])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test(im_path="test_img.jpg")
