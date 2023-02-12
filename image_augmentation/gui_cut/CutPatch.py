import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from tkinter import filedialog
import os

root_im_dir = "DB"
nolabel_dir = "NL"

# TK instance
root = tk.Tk()
root.title("CatPatch 🐱")

# Window dimensions
root.geometry("196x170")

# Frames
frame_select = tk.Frame(root, relief="groove", borderwidth=3)
frame_select.grid(row=0, column=0, columnspan=3)

frame_cut = tk.Frame(root, relief="groove", borderwidth=3)
frame_cut.grid(row=1, column=0, columnspan=3)


# Load the selected image
select_title = tk.Label(frame_select, text="Raw image")
select_title.grid(row=0, column=0, columnspan=2)
label_title = tk.Label(frame_select, text="Label:")
label_title.grid(row=1, column=1, padx=(18, 0))
label = tk.StringVar()
label.set("ED")
label_entry = tk.Entry(frame_select, width=8, textvariable=label)
label_entry.grid(row=1, column=2)
filepath = None
ax = None
image = None
H, W = 0, 0
def select_image():
    global filepath, ax, image
    global H, W
    if filepath := filedialog.askopenfilename():
        fig, ax = plt.subplots()
        image = plt.imread(filepath)
        H, W, _ = np.shape(image)
        plt.imshow(image)
        plt.xlim([0, W])
        plt.ylim([0, H])
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
load_button = tk.Button(frame_select, text="Load", command=select_image)
load_button.grid(row=1, column=0)


# Cut the image into sub-images using the entered dimensions
cut_title = tk.Label(frame_cut, text="Patches     ")
cut_title.grid(row=0, column=0, columnspan=2)
entry_width = tk.Entry(frame_cut, width=8)
entry_width.grid(row=1, column=0)
label_sep = tk.Label(frame_cut, text="x")
label_sep.grid(row=1, column=1)
entry_height = tk.Entry(frame_cut, width=8)
entry_height.grid(row=1, column=2)

step_w, step_h = None, None
patches = []
lines = []
def cut_image():
    try:
        global step_w, step_h
        step_w = int(entry_width.get())
        step_h = int(entry_height.get())
    except ValueError:
        pass
    else:
        if step_w > 0 and step_h > 0:
            global ax, H, W, patches
            # cleaning
            ax.lines.clear()
            [patch.remove() for patch in patches if patch]

            w, step_w = np.linspace(0, W, int(W/step_w)+1, retstep=True)
            h, step_h = np.linspace(0, H, int(H/step_h)+1, retstep=True)
            patches = [None] * ((len(w)-1) * (len(h)-1))

            for y in h: ax.axhline(y, color='r', linestyle='-')
            for x in w: ax.axvline(x, color='r', linestyle='-')
            plt.draw()
cut_button = tk.Button(frame_cut, text="Cut", command=cut_image)
cut_button.grid(row=2, column=1)


# Manage user clic to (dis)enable patches
def onclick(event):
    global patches
    try:
        x, y = int(event.xdata), int(event.ydata)
    except TypeError:
        pass
    else:
        pos_w = int(x/step_w)
        pos_h = int(y/step_h)
        N = pos_h * int(W/step_w) + pos_w
        if patches[N]:
            patches[N].remove()
            patches[N] = None
        else:
            start_x = pos_w * step_w
            start_y = pos_h * step_h
            rect = Rectangle((start_x, start_y), step_w, step_h, color='gray')
            ax.add_patch(rect)
            patches[N] = rect
        plt.draw()


# Save the patches in a selected folder
def save_images():
    global step_h, step_w
    if not filepath:
        return
    fname, _ = os.path.splitext(os.path.basename(filepath))
    os.makedirs(root_im_dir, exist_ok=True)
    os.makedirs(f"{root_im_dir}/{label.get()}", exist_ok=True)
    os.makedirs(f"{root_im_dir}/{nolabel_dir}", exist_ok=True)
    step_h = int(step_h)
    step_w = int(step_w)
    for N, patch in enumerate(patches):
        pos_h = int(N / int(W/step_w))
        pos_w = N - pos_h * int(W/step_w)
        y = pos_h * step_h
        x = pos_w * step_w
        sub_image = np.flipud(image[y:y+step_h, x:x+step_w, :])
        if patch:
            patch_path = f"{root_im_dir}/{nolabel_dir}/{fname}_{N}.jpg"
        else:
            patch_path = f"{root_im_dir}/{label.get()}/{fname}_{N}.jpg"
        Image.fromarray(sub_image).save(patch_path, format="JPEG", quality=100))

save_button = tk.Button(root, text="Save", command=save_images)
save_button.grid(row=2, column=1)


root.mainloop()
