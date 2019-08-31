import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preconvert.output import json
import tabulate
import tempfile


class BaseCompressor:
    bench_exceptions = ()
    compression_options = []

    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name

    def infer(self, obj):
        raise NotImplementedError

    def _save(self, obj, fname, *args, **kwargs):
        raise NotImplementedError

    def save(self, obj, fname_prefix, inferred_kwargs=None):
        if inferred_kwargs is None:
            inferred_kwargs = self.infer(obj)
        return self._save(obj, fname_prefix, **inferred_kwargs)

    def load(self, path):
        raise NotImplementedError

    def get_features(self, *args, **kwargs):
        raise NotImplementedError

    def is_valid(self, *args, **kwargs):
        raise NotImplementedError

    def benchmark(self, object_save, kwargs):
        t1 = time.time()
        path = "ZZZklkl"
        with tempfile.TemporaryDirectory() as fdir:
            try:
                path = self.save(object_save, fdir + "/", kwargs)
            except self.bench_exceptions:
                return None, None, None
            try:
                write_time = time.time() - t1
                size = os.path.getsize(path)
                t1 - time.time()
                _ = self.load(path)
                read_time = time.time() - t1
            except self.bench_exceptions:
                return None, None, None
        return size, write_time, read_time

    def run_benchmarks(self, data_generator, save=True, start_from_beginning=True):
        model_path = os.path.expanduser("~/shrynk_{}.jsonl".format(self.model_name))
        num_skip = -1
        if not start_from_beginning and os.path.isfile(model_path):
            with open(model_path) as f:
                num_skip = len(f.read().split("\n")) // len(self.compression_options)
        results = []
        index = []
        for num, df in enumerate(data_generator):
            if num < num_skip:
                continue
            if isinstance(df, str) and os.path.isfile(df):
                try:
                    df = self.load(df)
                except self.bench_exceptions:
                    continue
            if df is None:
                continue
            if isinstance(df, str):
                continue
            stat_computation_time = time.time()
            features = self.get_features(df)
            stat_computation_time = time.time() - stat_computation_time
            group_id = hash(str(features))
            if features is None:
                continue
            for kwargs in self.compression_options:
                size, write_time, read_time = self.benchmark(df, kwargs)
                print(kwargs, size, write_time, read_time)
                if size is None:
                    # write_error(line)
                    continue
                # print("   ", size, write_time, read_time, engine, compression)
                result = {"size": size, "write_time": write_time, "read_time": read_time}
                jdata = {
                    "group_id": group_id,
                    "class": self.__class__.__name__,
                    "kwargs": kwargs,
                    "features": features,
                    "stat_computation_time": stat_computation_time,
                    "result": result,
                }
                if save:
                    with open(model_path, "a") as f:
                        f.write(json.dumps(jdata) + "\n")
                index.append(" ".join(["{}={!r}".format(k, v) for k, v in kwargs.items()]))
                results.append(result)
        info = pd.DataFrame(results, index=index, columns=["size", "write_time", "read_time"])
        return info

    def show_score(self, info, size_weight=100, write_weight=-10, read_weight=-1):
        info["score"] = (info["size"] * size_weight) / -(
            (info["write_time"] * write_weight) * (info["read_time"] * read_weight)
        )
        print(tabulate.tabulate(info.sort_values("score"), headers=info.columns))
        del info["score"]
