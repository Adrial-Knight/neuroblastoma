from pathlib import Path
import sys

ACCOUNTS = ["Un Kiwi", "Adrial-Knight", "P.J.M.", "P.M. matmeca", "P.M."] \
         + [f"P.C. {d:02d}" for d in range(1, 36)]

HISTORIC = "./historic.txt"

def get_historic():
    return Path(HISTORIC).read_text().splitlines()

def check(historic):
    for l, account in enumerate(historic):
        if not account in ACCOUNTS:
            print(f"[l. {l}]: {account} not allowed")
            return False
    return True

def get_unused(ACCOUNTS, historic, nb=3):
    if len(ACCOUNTS) < nb+1:
        return ACCOUNTS
    historic[:0] = ACCOUNTS
    while len(ACCOUNTS) != nb:
        account = historic.pop()
        historic = list(filter((account).__ne__, historic))
        ACCOUNTS = list(filter((account).__ne__, ACCOUNTS))
    return ACCOUNTS

if __name__ == "__main__":
    historic = get_historic()
    if not check(historic):
        exit(2)
    else:
        try:
            if len(sys.argv) == 2:
                nb = int(sys.argv[1])
            else: raise ValueError
        except ValueError:
            nb = 1
        unused = get_unused(ACCOUNTS, historic, nb)
        for item in unused:
            print(item, end=" ")
        print("", end="\n")
