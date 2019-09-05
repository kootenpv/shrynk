import time
import os
import pandas as pd
import tempfile
from shrynk.utils import md5, get_model_data


from wrapt_timeout_decorator import timeout as timeout_fn


class BaseCompressor:
    bench_exceptions = ()
    compression_options = []

    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name
        self.model_data = None

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

    def benchmark(self, object_save, kwargs, timeout):
        t1 = time.time()
        path = "ZZZklkl"
        with tempfile.TemporaryDirectory() as fdir:
            try:
                if timeout:
                    path = timeout_fn(timeout)(self.save)(object_save, fdir + "/", kwargs)
                else:
                    path = self.save(object_save, fdir + "/", kwargs)
            except TimeoutError:
                return None, None, None
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

    def run_benchmarks(self, data_generator, save=True, ignore_seen=True, timeout=300):
        from preconvert.output import json

        model_path = os.path.expanduser("~/shrynk_{}.jsonl".format(self.model_name))
        if self.model_data is None:
            self.model_data = get_model_data(self.model_name, self.compression_options)
        feature_ids = set([x["feature_id"] for x in self.model_data])
        results = []
        index = []
        if isinstance(data_generator, pd.DataFrame):
            data_generator = [data_generator]
        for num, df in enumerate(data_generator):
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
            if features is None:
                continue
            feature_id = md5(features)
            if ignore_seen and feature_id in feature_ids:
                print("seen", feature_id)
                continue
            stat_computation_time = time.time() - stat_computation_time
            result = {
                "feature_id": feature_id,
                "features": features,
                "class": self.__class__.__name__,
                "stat_computation_time": stat_computation_time,
            }
            bench = []
            for kwargs in self.compression_options:
                size, write_time, read_time = self.benchmark(df, kwargs, timeout)
                if size is None:
                    # write_error(line)
                    print("error, skipping", kwargs)
                    continue
                print(kwargs, size, write_time, read_time)
                bench.append(
                    {
                        "kwargs": json.dumps(kwargs),
                        "size": size,
                        "write_time": write_time,
                        "read_time": read_time,
                    }
                )
            result["bench"] = bench
            if bench:
                self.model_data.append(result)
                if save:
                    with open(model_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                results.append(result)

            feature_ids.add(feature_id)
        ### run benchmarks should return a total overview or something, but now just from the last df
        # return pd.DataFrame(bench).set_index("kwargs")
        return results
