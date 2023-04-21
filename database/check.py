from PIL import Image
import os
import sys

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_image_sizes(db):
    categories = os.listdir(db)
    list_size = set()
    for cat in categories:
        imfnames = os.listdir(f"{db}/{cat}")
        for i, im in enumerate(imfnames, start=1):
            with Image.open(f"{db}/{cat}/{im}") as img:
                W, H = img.size
                list_size.add((W, H))
    return sorted(list_size)


def cut_proposal(list_size, w, h):
    print(f"Cut for {w, h} patches")
    for size in list_size:
        W = int(size[0] / w) * w
        H = int(size[1] / h) * h
        print(f"{size} --> {(W, H)}")

def what_image_not_pre_scaled(db, w, h):
    categories = os.listdir(db)
    res = True
    for cat in categories:
        imfnames = os.listdir(f"{db}/{cat}")
        for i, im in enumerate(imfnames, start=1):
            with Image.open(f"{db}/{cat}/{im}") as img:
                W, H = img.size
                expected_W = w * (W // w)
                expected_H = h * (H // h)
                if expected_W != W or expected_H != H :
                    print(f"{im}: {(W, H)} --> {(expected_W, expected_H)}")
                    res = False
    if res: print(bcolors.OKGREEN, end="")
    else: print(bcolors.FAIL, end="")
    print("Image pre-scaled" + bcolors.ENDC)

def is_image_scaled(db, w, h):
    categories = os.listdir(db)
    res = True
    for cat in categories:
        try:
            imfnames = os.listdir(f"{db}/{cat}")
        except NotADirectoryError: continue
        for i, im in enumerate(imfnames, start=1):
            with Image.open(f"{db}/{cat}/{im}") as img:
                W, H = img.size
                if W != w or H != h:
                    print(f"{im}: {(W, H)} --> {(w, h)}")
                    res = False
    if res: print(bcolors.OKGREEN, end="")
    else: print(bcolors.FAIL, end="")
    print("Image scaled" + bcolors.ENDC)

def is_fname_agreed_catname(db):
    categories = os.listdir(db)
    res = True
    for cat in categories:
        if cat == "NL": continue  # No Label
        try:
            imfnames = os.listdir(f"{db}/{cat}")
        except NotADirectoryError: continue
        len_cat = len(cat)
        for fname in imfnames:
            if fname[:len_cat] != cat:
                print(f"{cat} contains {fname}")
                res = False
    if res: print(bcolors.OKGREEN, end="")
    else: print(bcolors.FAIL, end="")
    print("Image in right cat" + bcolors.ENDC)

def print_stat(db):
    categories = os.listdir(db)
    total_pix = 0
    total_img = 0
    for cat in categories:
        if cat == "G": continue
        try:
            imfnames = os.listdir(f"{db}/{cat}")
        except NotADirectoryError: continue
        sum_pix  = 0
        num_img  = len(imfnames)
        for im in imfnames:
            with Image.open(f"{db}/{cat}/{im}") as img:
                W, H = img.size
                sum_pix += W*H
        print(f"{cat}: {sum_pix=}, {num_img=}")
        total_pix += sum_pix
        total_img += num_img
    print(f"{total_pix=}, {total_img=}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 check.py db_wxh")
        exit(1)

    dbname = sys.argv[1]
    if dbname[-1] == '/': dbname = dbname[:-1]
    if "x" in dbname:
        dimensions = dbname.split("_")[0]
        w, h = dimensions.split("x")
        w, h = int(w), int(h)
        is_fname_agreed_catname(dbname)
        if "crop_for" in dbname:
            print_stat(dbname)
            # list_size = get_image_sizes(db)
            # cut_proposal(list_size, w=dim, h=dim)
            what_image_not_pre_scaled(dbname, w, h)
        else:
            is_image_scaled(dbname, w, h)
