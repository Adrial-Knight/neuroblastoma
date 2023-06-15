import os
import time
import datetime
import json
import matplotlib.pyplot as plt
from PIL import Image
import logging
from flask import Flask, request, send_file, render_template, url_for, redirect
from googleapiclient.errors import HttpError

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import drive.goolgeapiclient_wrap as Gdrive
import drive.grid as GridSearch
import drive.main as GridMetric
import drive.alive as AccountChecker
import drive.custom_sort as CustomSort
import drive.available as AccountHistoric

app = Flask(__name__)
app.secret_key = "secret_key"
session = {}
drive = Gdrive.identification()
PORT = 5000
HOST = "localhost"
TO_SKIP = {"models": ["__Results__", "LeakTest"]}
ROOT_PATH = "Stage_Bilbao_Neuroblastoma/G_Collab"
__is_server_busy__ = False

@app.route("/")
def index():
    all_models = list_models()
    notebooks = AccountChecker.get_notebook_ids(drive, ROOT_PATH)
    session["forced_update"] = False
    session["all_models"] = all_models
    session["notebooks"] = notebooks
    session["busy_accounts"] = AccountChecker.find_busy_accounts(drive, notebooks, delay=120, utc_offset=2)
    session["next_accounts"] = AccountHistoric.get_unused_for_dashboard(len(notebooks))
    session["date"] = datetime.datetime.now().strftime("%H:%M")
    return render_template(
        "index.html",
        busy_accounts=len(session["busy_accounts"]),
        notebooks=len(session["notebooks"]),
        next_accounts = session["next_accounts"],
        date=session["date"],
        all_models=session["all_models"])

@app.route("/display", methods=["GET", "POST"])
def display_static():
    if request.method == "POST":
        session["model"] = request.form["model"]
    model = session.get("model", None)

    gridsearch = f"fig/{model}_gridsearch.png"
    gridloss = f"fig/{model}_lossGridMetrics.png"
    gridaccu = f"fig/{model}_accuGridMetrics.png"

    need_update = False
    for grid in [gridsearch, gridloss, gridaccu]:
        if not os.path.exists(f"static/{grid}"):
            need_update = True
            create_tmp_img(f"static/{grid}")
    if need_update:
        return redirect("/update")
    else:
        best_loss, best_accu = get_best_metrics()
        return render_template(
            "index.html",
            model=model,
            all_models=session["all_models"],
            busy_accounts=len(session["busy_accounts"]),
            notebooks=len(session["notebooks"]),
            next_accounts=session["next_accounts"],
            date=session["date"],
            best_loss=best_loss,
            best_accu=best_accu,
            gridsearch=url_for("static", filename=gridsearch),
            gridloss=url_for("static", filename=gridloss),
            gridaccu=url_for("static", filename=gridaccu)
        )

@app.route("/update")
def update():
    model = session.get("model", None)
    if not model:
        return redirect("/")
    elif not __is_server_busy__:
        lock_server()
        try:
            notebooks = session["notebooks"]
            if session["forced_update"]:
                grid_metric_delay = 0
                session["forced_update"] = False
            else:
                grid_metric_delay = 3600
            update_grid_search(model, delay=0)
            if data := update_grid_metric(model, delay=grid_metric_delay):
                update_best_metrics(data)
            try:
                session["busy_accounts"] = AccountChecker.find_busy_accounts(drive, notebooks, delay=120, utc_offset=2)
            except HttpError as e:
                _print("Change in some notebook IDs")
                notebooks = AccountChecker.get_notebook_ids(drive, ROOT_PATH)
                session["notebooks"] = notebooks
                session["busy_accounts"] = AccountChecker.find_busy_accounts(drive, notebooks, delay=120, utc_offset=2)

            session["date"] = datetime.datetime.now().strftime("%H:%M")
            session["next_accounts"] = AccountHistoric.get_unused_for_dashboard(len(session["notebooks"]))
        except HttpError as e:
            _print(e)
        finally:
            unlock_server()
    return redirect("/display")

@app.route("/forced_update")
def forced_update():
    session["forced_update"] = True
    return redirect("/update")

@app.route("/reload")
def reload():
    return update()

def lock_server():
    global __is_server_busy__
    if not __is_server_busy__:
        __is_server_busy__ = True
        _print("busy")
    else:
        _print("...")

def unlock_server():
    global __is_server_busy__
    if __is_server_busy__:
        __is_server_busy__ = False

def _print(message: str):
    now = datetime.datetime.now().strftime("[%d/%b/%Y %H:%M:%S]")
    print(f"{HOST}:{PORT} - - {now} \"{message}\"")

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

def update_grid_search(model, delay):
    path = f"static/fig/{model}_gridsearch.png"
    if not os.path.exists(path) \
    or time.time() - os.path.getmtime(path) > delay:
        grid, active = GridSearch.make_grid(drive, model)
        fig = GridSearch.display_grid(grid, active, model)
        fig.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

def update_grid_metric(model, delay):
    path = {"loss": f"static/fig/{model}_lossGridMetrics.png",
            "accu": f"static/fig/{model}_accuGridMetrics.png"}
    if not os.path.exists(path["loss"]) \
    or time.time() - os.path.getmtime(path["loss"]) > delay:
        data, _ = GridMetric.update_json_tab(drive, f"{ROOT_PATH}/backup/{model}")
        for metric in ["accu", "loss"]:
            fig = GridMetric.merge_cell_metric(data, metric)
            fig.savefig(path[metric], facecolor=fig.get_facecolor(), edgecolor='none')
        return data
    else:
        return None

def update_best_metrics(data):
    best = {"loss": float("inf"), "accu": 0}
    for key, val in data.items():
        best["loss"] = min(best["loss"], min(val["best_loss"]))
        best["accu"] = max(best["accu"], max(val["best_accu"]))
    with open("static/json/best_metrics.json", "r") as fd:
        data = json.load(fd)
        data[session["model"]] = best
    with open("static/json/best_metrics.json", "w") as fd:
        json.dump(data, fd)

def get_best_metrics():
    model = session.get("model", None)
    loss = None
    accu = None
    if model:
        with open("static/json/best_metrics.json", "r") as fd:
            data = json.load(fd)
            if model in data.keys():
                loss = round(data[model]["loss"], 3)
                accu = round(data[model]["accu"] * 100, 2)
    return loss, accu

def create_tmp_img(path):
    Image.new("RGB", (16, 16), "white").save(path)

def list_models():
    root_id = Gdrive.get_id_from_path(drive, path=f"{ROOT_PATH}/backup")
    models, _ = Gdrive.list_from_id(drive, root_id)
    for folder in TO_SKIP["models"]:
        models.remove(folder)
    models = sorted(models, key=CustomSort.alphanum)
    return models


if __name__ == "__main__":
    if len(sys.argv) == 2: PORT = int(sys.argv[1])

    # Masquer les sorties de Flask dans le terminal
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host=HOST, port=PORT, debug=False)
