import time

try:
    from . import goolgeapiclient_wrap as Gdrive
except ImportError:
    import goolgeapiclient_wrap as Gdrive


def find_busy_accounts(drive, notebook_ids, delay=90, utc_offset=2):
    now = time.time()
    accounts = []
    for id in notebook_ids:
        infos = Gdrive.get_file_infos_from_id(drive, id, utc_offset)
        if abs(now - infos["modifiedTimeSec"]) < delay:
            accounts.append(infos["lastModifyingUser"])
    return accounts

def get_notebook_ids(drive, root_path):
    root_id = Gdrive.get_id_from_path(drive, root_path)
    title_list, id_list = Gdrive.list_from_id(drive, root_id)
    offset = 0
    for i, title in enumerate(title_list):
        if not title.startswith("Training #"):
            id_list.pop(i - offset)
            offset += 1
    return id_list



if __name__ == "__main__":
    root = "Stage_Bilbao_Neuroblastoma/G_Collab/"
    drive = Gdrive.identification()
    notebook_ids = get_notebook_ids(drive, root)
    print(notebook_ids)
    accounts = find_busy_accounts(drive, notebook_ids, delay=90, utc_offset=2)
    print(accounts)
