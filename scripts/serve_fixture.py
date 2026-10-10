from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from grid_forecast.data import NODES
from grid_forecast.experiments import run_experiment
from grid_forecast.io import file_hash, write_json


def build_fixture(root):
    root = Path(root); data = root/"data"; data.mkdir(parents=True)
    index = pd.date_range("2012-01-01T01:00", periods=2000, freq="h")
    time = np.arange(len(index))
    leaves = np.column_stack([(i+1)*100+20*np.sin(time/24)+np.random.default_rng(i).normal(0,2,len(time)) for i in range(5)])
    values = np.column_stack([leaves, leaves.sum(axis=1)])
    pd.DataFrame({"timestamp":np.repeat(index,6),"node":np.tile(NODES,len(index)),"demand_mw":values.reshape(-1)}).to_parquet(data/"demand.parquet",index=False)
    write_json(data/"manifest.json",{"parquet_sha256":file_hash(data/"demand.parquet")})
    config = {"folds":["2012-03-01T00:00:00"],"test_days":2,"validation_days":20,"calibration_days":10,
              "training_hours":720,"training_stride":6,"evaluation_stride":6,"horizons":[1,6,24,168],
              "quantiles":[.1,.5,.9],"candidates":[{"max_depth":2,"n_estimators":8}],"learning_rate":.07,
              "seed":42,"seeds":[42],"threads":1,"coverage":.8,
              "neural":{"history_hours":336,"patch_hours":24,"d_model":8,"heads":2,"layers":1,"dropout":.1,
                        "epochs":1,"batch_size":32,"learning_rate":.001,"weight_decay":.0001}}
    output = root/"experiment"; run_experiment(data,config,output)
    return data,output


if __name__ == "__main__":
    import uvicorn
    from grid_forecast.service import create_app
    with tempfile.TemporaryDirectory(prefix="grid-browser-") as directory:
        data,experiment=build_fixture(directory)
        uvicorn.run(create_app(data,experiment,Path(__file__).resolve().parents[1]/"frontend/dist"),host="127.0.0.1",port=8013)
