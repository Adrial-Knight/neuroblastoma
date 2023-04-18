import pydrive_wrap as Gdrive

SUMMARY_FOLDER = "__Summary__"
RESULTS_FOLDER = "__Results__"
SKIP_FOLDER = ["LeakTest", RESULTS_FOLDER]


def main(backup_path):
    drive = Gdrive.identification()
    backup_id = Gdrive.get_id_from_path(drive, backup_path)
    network_titles, network_ids = Gdrive.list_from_id(drive, backup_id)

    def extract_result_network(network_id):
        sum_folder_id = Gdrive.get_id_from_folder_id(drive, network_id, SUMMARY_FOLDER)
        data_id = Gdrive.get_id_from_folder_id(drive, sum_folder_id, "data.json")
        data = Gdrive.load_json_from_id(drive, data_id)
        Gdrive.to_json_compatible(data)

        best_loss = 1000
        best_accu = 0
        result = {"loss":
                    {"best": 1000, "epoch": -1, "lr": -1, "batch": -1, "exp": -1},
                  "accu":
                    {"best": 0.00, "epoch": -1, "lr": -1, "batch": -1, "exp": -1}}
        for key, val in data.items():
            exp_l, loss = min(enumerate(val["best_loss"]), key=lambda x: x[1])
            exp_a, accu = max(enumerate(val["best_accu"]), key=lambda x: x[1])
            lr, batch = key.split("_")
            lr = float(lr)
            batch = int(batch)
            if loss < result["loss"]["best"]:
                result["loss"]["best"]  = loss
                result["loss"]["epoch"] = val["best_epoch_loss"][exp_l]
                result["loss"]["lr"] = lr
                result["loss"]["batch"] = batch
                result["loss"]["exp"] = exp_l + 1
            if accu > result["accu"]["best"]:
                result["accu"]["best"]  = accu
                result["accu"]["epoch"] = val["best_epoch_loss"][exp_a]
                result["accu"]["lr"] = lr
                result["accu"]["batch"] = batch
                result["accu"]["exp"] = exp_a + 1
        return result

    data = {}
    for title, id in zip(network_titles, network_ids):
        if title in SKIP_FOLDER: continue
        result = extract_result_network(id)
        data[title] = result
        print(f"{title}")
        print(f"\tLoss={result['loss']['best']:.3f}  (epoch: {int(result['loss']['epoch']):2d}) | (lr, batch)=({result['loss']['lr']}, {result['loss']['batch']:3d}) at exp {result['loss']['exp']}")
        print(f"\tAccu={result['accu']['best']:.2%} (epoch: {int(result['accu']['epoch']):2d}) | (lr, batch)=({result['accu']['lr']}, {result['accu']['batch']:3d}) at exp {result['accu']['exp']}")
    results_id = network_ids[network_titles.index(RESULTS_FOLDER)]
    Gdrive.save_dic_to_drive(drive, data, "result.json", results_id)

if __name__ == "__main__":
    path = "Stage_Bilbao_Neuroblastoma/G_Collab/backup"
    main(path)
