from pathlib import Path
import sys

ACCOUNTS = ["Un Kiwi", "Adrial-Knight", "P.J.M.", "P.M. matmeca", "P.M. enseirb", "P.M.", "P.M. 58"] \
         + [f"P.C. {d:02d}" for d in range(1, 58)]

BANNED = ["P.C. 47"]

def get_historic(historic_path):
    return Path(historic_path).read_text().splitlines()

def check(historic):
    for l, account in enumerate(historic):
        if not account in ACCOUNTS and not account in BANNED:
            print(f"[l. {l}]: {account} not allowed")
            return False
    return True

def get_unused(ACCOUNTS, historic, nb=3):
    ACCOUNTS = list(set([account for account in ACCOUNTS if account not in BANNED]))
    historic = ACCOUNTS + historic
    used, unused = [], []
    for account in historic[::-1]:
        if account in BANNED:
            continue
        used.append(account)
        if len(used) == len(ACCOUNTS) - nb:
            unused = [account for account in historic[::-1] if account not in used][:nb][::-1]
            break
    return unused

def get_unused_for_dashboard(nb):
    historic = get_historic("../drive/historic.txt")
    unused = get_unused(ACCOUNTS, historic, nb)
    for i, account in enumerate(unused):
        if account.startswith("P.C. "):
            id = account.split("P.C. ")[-1]
            account = f"pierre.colab.{id}@gmail.com"
        elif account.startswith("P.M. "):
            id = account.split("P.M. ")[-1]
            account = f"pierre.minier.{id}@gmail.com"
        elif account == "Un Kiwi": account = "avion.bleu.64@gmail.com"
        elif account == "Adrial-Knight": account = "adrial.knight@gmail.com"
        elif account == "P.J.M.": account = "p.minier@opendeusto.es"
        elif account == "P.M.": account="pierre.minier.64@gmail.com"
        unused[i] = account
    return unused


if __name__ == "__main__":
    historic = get_historic("./historic.txt")
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
