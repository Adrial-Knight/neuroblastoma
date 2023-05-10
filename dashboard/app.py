import os
import time
import json
import matplotlib.pyplot as plt
from flask import Flask, request, send_file, render_template, url_for, redirect, session

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import drive.goolgeapiclient_wrap as Gdrive
import drive.grid as GridSearch
import drive.main as GridMetric
import drive.alive as AccountChecker

app = Flask(__name__)
app.secret_key = "secret_key"
drive = Gdrive.identification()
TO_SKIP = {"models": ["__Results__", "LeakTest"]}
ROOT_PATH = "Stage_Bilbao_Neuroblastoma/G_Collab"
__is_server_busy__ = False

@app.route("/")
def index():
    all_models = list_models()
    notebooks = AccountChecker.get_notebook_ids(drive, ROOT_PATH)
    session["all_models"] = all_models
    session["notebooks"] = notebooks
    session["accounts"] = AccountChecker.find_busy_accounts(drive, notebooks, delay=90, utc_offset=2)
    return render_template(
        "index.html",
        accounts=len(session["accounts"]),
        notebooks=len(session["notebooks"]),
        all_models=session["all_models"])

@app.route("/display", methods=["GET", "POST"])
def display_static():
    if request.method == "POST":
        session["model"] = request.form["model"]
    model = session.get("model", None)

    gridsearch = f"fig/{model}_gridsearch.png"
    gridloss = f"fig/{model}_lossGridMetrics.png"
    gridaccu = f"fig/{model}_accuGridMetrics.png"

    if not os.path.exists(f"static/{gridsearch}") \
    or not os.path.exists(f"static/{gridloss}")   \
    or not os.path.exists(f"static/{gridaccu}"):
        return redirect("/update")

    else:
        return render_template(
            "index.html",
            model=model,
            all_models=session["all_models"],
            accounts=len(session["accounts"]),
            notebooks=len(session["notebooks"]),
            gridsearch=url_for("static", filename=gridsearch),
            gridloss=url_for("static", filename=gridloss),
            gridaccu=url_for("static", filename=gridaccu)
        )

@app.route("/update")
def update():
    if not __is_server_busy__:
        lock_server()
        model = session.get("model", None)
        notebooks = session["notebooks"]
        update_grid_search(model, delay=1)
        update_grid_metric(model, delay=1)
        session["accounts"] = AccountChecker.find_busy_accounts(drive, notebooks, delay=90, utc_offset=2)
        print(f"Accounts: {session['accounts']}")
        unlock_server()
    return redirect("/display")

@app.route("/reload")
def reload():
    return update()

def lock_server():
    global __is_server_busy__
    if not __is_server_busy__:
        __is_server_busy__ = True
        print("[SERVER]: Locked")
    else:
        print("[SERVER]: Busy")

def unlock_server():
    global __is_server_busy__
    if __is_server_busy__:
        __is_server_busy__ = False
        print("[SERVER]: Unlocked")

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

def update_grid_search(model, delay):
    path = f"static/fig/{model}_gridsearch.png"
    if not os.path.exists(path) \
    or time.time() - os.path.getmtime(path) > delay:
        grid = GridSearch.make_grid(drive, model)
        fig = GridSearch.display_grid(grid, model)
        fig.savefig(path, bbox_inches='tight', pad_inches=0)

def update_grid_metric(model, delay):
    path = {"loss": f"static/fig/{model}_lossGridMetrics.png",
            "accu": f"static/fig/{model}_accuGridMetrics.png"}
    if not os.path.exists(path["loss"]) \
    or time.time() - os.path.getmtime(path["loss"]) > delay:
        data, _ = GridMetric.update_json_tab(drive, f"{ROOT_PATH}/backup/{model}")
        for metric in ["accu", "loss"]:
            fig = GridMetric.merge_cell_metric(data, metric)
            fig.savefig(path[metric])

def list_models():
    root_id = Gdrive.get_id_from_path(drive, path=f"{ROOT_PATH}/backup")
    models, _ = Gdrive.list_from_id(drive, root_id)
    for folder in TO_SKIP["models"]:
        models.remove(folder)
    return models


if __name__ == "__main__":
    app.run()
