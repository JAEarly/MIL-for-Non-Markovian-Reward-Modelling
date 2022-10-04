import wandb
from tqdm import tqdm
from numpy import array, transpose
from pandas import DataFrame


def get(project_name, metrics, filters=None, tag=None):
    data = [{"run_names": [], "metrics": []} for _ in filters]
    # TODO: Can apply filters directly to api.runs
    for run in tqdm(wandb.Api().runs(project_name)):
        if tag is None or tag in run.tags:
            config = run.config; config.update({"name": run.name})
            active_filters = set()
            for i, f in enumerate(filters):
                activate = True
                for k, v_filter in f.items():
                    v = config; invalid = False
                    for k_sub in k.split("."):
                        try: v = v[k_sub]
                        except: invalid = True; break
                    if invalid or v != v_filter: activate = False; break
                if activate: active_filters.add(i)
            if active_filters:
                run_metrics = []
                for step in tqdm(run.scan_history(), leave=False):
                    if any("video." in m for m in step): continue # NOTE: Skip video logging steps
                    run_metrics.append([step[m] if m in step else float("nan") for m in metrics])
                for i in active_filters:
                    data[i]["run_names"].append(run.name)
                    data[i]["metrics"].append(run_metrics)    
    dataframes = [{} for _ in filters]
    for i, f in enumerate(filters):
        for m, data_m in zip(metrics, transpose(array(data[i]["metrics"]))):
            dataframes[i][m] = {
                "fname": "-".join([f"{k}={v}" for k,v in f.items()]) + "---" + m + ".csv",
                "df": DataFrame(data_m, columns=data[i]["run_names"])
            }
    return dataframes
