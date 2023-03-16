import json
import io
import numpy as np

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def identification():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def list_from_id(drive, id):
    query = f"'{id}' in parents and trashed=false"
    file_list = drive.ListFile({'q': query}).GetList()
    title_list = [file["title"] for file in file_list]
    id_list = [file["id"] for file in file_list]
    return title_list, id_list

def get_id_from_path(drive, path):
    folders = path.split("/")  # Split into individual folder names
    folder_id = "root"         # First folder has 'root' as ID

    # Traverse the folder hierarchy
    for i, folder_title in enumerate(folders):
        # Search for the folder by name and parent ID
        query = f"'{folder_id}' in parents and title='{folder_title}' and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()

        if len(file_list) == 0:
            parent_file_list = list_from_id(drive, folder_id)
            parent_name = folders[i-1] if i else "the root"
            print(f"{folder_title} not found in {parent_name}")
            print("Found:", end=" ")
            for file in parent_file_list[0]:
                print(file, end="    ")
            print("", end="\n")
            return None
        else:
            folder_id = file_list[0]["id"]  # update the folder ID

    return folder_id

def load_json_from_id(drive, id):
    if not id: return None
    file = drive.CreateFile({'id': id})
    if file['mimeType'] != 'application/json':
        raise ValueError(f"File with ID {id} is not a JSON file")
    else:
        content = file.GetContentString()
        return json.loads(content)

def save_dic_to_drive(drive, data, fname, folder_id):
    query = f"'{folder_id}' in parents and trashed = false and title = '{fname}'"
    files = drive.ListFile({'q': query}).GetList()
    if len(files) == 1:
        file = files[0]
    else:
        file_metadata = {"title": fname, "parents": [{"kind": "drive#fileLink", "id": folder_id}]}
        file = drive.CreateFile(file_metadata)
    file.content = io.BytesIO(json.dumps(to_json_compatible(data)).encode())
    file.Upload()
    return file["id"]

def to_json_compatible(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: to_json_compatible(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_json_compatible(v) for v in data]
    elif np.isnan(data):
        return "NaN"
    else:
        return data

def from_json_compatible(data):
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: from_json_compatible(v) for k, v in data.items()}
    else:
        return data

def upload_fig(drive, folder_id, fig, fname):
    fullname = f"{fname}.pdf"
    tmp_local = f"/tmp/{fullname}"
    fig.savefig(tmp_local)

    query = f"'{folder_id}' in parents and title = '{fullname}' and trashed = false"
    file_list = drive.ListFile({'q': query}).GetList()

    if len(file_list) == 1:  # Fichier déjà existant => mise à jour
        file = file_list[0]
    else: # Aucun fichier => création
        file = drive.CreateFile({'title': fullname, 'parents': [{'id': folder_id}]})
    file.SetContentFile(tmp_local)
    file.Upload()
    return file["id"]
