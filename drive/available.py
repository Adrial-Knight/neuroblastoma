from pathlib import Path

ACCOUNTS = ["Un Kiwi", "Adrial-Knight", "P.J.M.", "P.M. matmeca", "P.M."] \
         + [f"P.C. {d:02d}" for d in range(1, 26)]

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
    if len(ACCOUNTS) < nb+1: exit(3)
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
        unused = get_unused(ACCOUNTS, historic, nb=1)
        print(f"{unused=}")
