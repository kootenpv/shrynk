import os
from gzip import GzipFile, decompress
import pkgutil
import hashlib
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
import functools


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


@functools.lru_cache()
def get_model_data(model_name):
    from preconvert.output import json

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
