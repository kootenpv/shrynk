import os
import pandas as pd
from gzip import GzipFile, decompress
import pkgutil
import hashlib
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale


def md5(features):
    from preconvert.output import json

    return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()


scalers = {
    "z": scale,
    "scale": scale,
    "minmax_scale": minmax_scale,
    "maxabs_scale": maxabs_scale,
    "robust_scale": robust_scale,
}

data_cache = {}


def get_model_data(model_name, compression_options=None):
    from preconvert.output import json

    if model_name in data_cache:
        return data_cache[model_name]
    """ Gets the model data """
    try:
        data = pkgutil.get_data("data", "shrynk/{}.jsonl.gzip".format(model_name.lower()))
        data = [
            json.loads(line) for line in decompress(data).decode("utf8").split("\n") if line.strip()
        ]
        print("from package")
    except FileNotFoundError:
        with open(os.path.expanduser("~/shrynk_{}.jsonl".format(model_name))) as f:
            data = [json.loads(x) for x in f.read().split("\n") if x]
    if compression_options is not None:
        known_kwargs = set([json.dumps(x) for x in compression_options])
        for x in data:
            x["bench"] = [y for y in x["bench"] if y["kwargs"] in known_kwargs]
        print("filtered compressions")
    data_cache[model_name] = data
    return data


def gzip(file_path):
    with open(file_path) as fin:
        with GzipFile(file_path + ".gzip", "w") as fout:
            fout.write(fin.read().encode("utf8"))


def gunzip(file_path):
    with GzipFile(file_path + ".gzip", "r") as fin:
        with open(file_path, "w") as fout:
            fout.write(fin.read())


def _package_data(file_path):
    file_path = os.path.expanduser(file_path)
    gzip(file_path)
    base = os.path.basename(file_path).replace("shrynk_", "")
    dest = "/home/pascal/egoroot/shrynk/data/shrynk/"
    os.rename(file_path + ".gzip", os.path.join(dest, base + ".gzip"))


def add_z_to_bench(bench, size, write, read, scaler="z"):
    if not isinstance(bench, pd.DataFrame):
        bench = pd.DataFrame(bench)[["kwargs", "size", "write_time", "read_time"]]
    if "kwargs" in bench.columns:
        bench = bench.set_index("kwargs")
    scale = scalers.get(scaler, scaler)
    z = (scale(bench) * (size, write, read)).sum(axis=1)
    bench["z"] = z
    return bench.sort_values("z")
