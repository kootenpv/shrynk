import time
import os
import pandas as pd
import tempfile
from shrynk.utils import md5, get_model_data, add_z_to_bench, shrynk_path


from wrapt_timeout_decorator import timeout as timeout_fn


class BaseCompressor:
    bench_exceptions = ()
    compression_options = []
    bench_base_path = "ZZZklkl"
    model_name = ""
    model_type = ""
    model_data = None

    def infer(self, obj):
        raise NotImplementedError

    def _save(self, obj, fname, *args, **kwargs):
        raise NotImplementedError

    def save(self, obj, fname_prefix, inferred_kwargs=None, allow_overwrite=False):
        if inferred_kwargs is None:
            inferred_kwargs = self.infer(obj)
        return self._save(obj, fname_prefix, allow_overwrite, **inferred_kwargs)

    def load(self, path):
        raise NotImplementedError

    def get_features(self, *args, **kwargs):
        raise NotImplementedError

    def is_valid(self, *args, **kwargs):
        raise NotImplementedError

    def single_benchmark(self, object_save, kwargs, timeout):
        t1 = time.time()
        path = self.bench_base_path
        with tempfile.TemporaryDirectory() as fdir:
            try:
                # TODO
                # XXX: make sure errors and timeouts also appear in the data
                if timeout:
                    path = timeout_fn(timeout)(self.save)(
                        object_save, fdir + "/" + self.bench_base_path, kwargs
                    )
                else:
                    path = self.save(object_save, fdir + "/" + self.bench_base_path, kwargs)
            except TimeoutError:
                return None, None, None
            except self.bench_exceptions as e:
                print("excp", e)
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

    def benchmark(self, df, timeout=300, verbose=False):
        from preconvert.output import json

        bench = []
        for kwargs in self.compression_options:
            size, write_time, read_time = self.single_benchmark(df, kwargs, timeout)
            if size is None:
                # write_error(line)
                print("error, skipping", kwargs)
                continue
            if verbose:
                print(kwargs, size, write_time, read_time)
            bench.append(
                {
                    "kwargs": json.dumps(kwargs),
                    "size": size,
                    "write_time": write_time,
                    "read_time": read_time,
                }
            )

        return bench

    def cast_to_data(self, df):
        if isinstance(df, str) and os.path.isfile(df):
            try:
                df = self.load(df)
            except self.bench_exceptions as e:
                print(e)
                return None, str(e)
        if df is None:
            return None, "df is None"
        if isinstance(df, str):
            return None, "df is str: " + df
        return df, "OK"

    def run_benchmarks(
        self, data_generator, ignore_seen=True, timeout=300, save=True, verbose=True
    ):
        from preconvert.output import json

        if self.model_data is None:
            self.model_data = get_model_data(
                self.model_type, self.model_name, self.compression_options
            )
        feature_ids = set([x["feature_id"] for x in self.model_data])
        results = []
        index = []
        if isinstance(data_generator, (str, pd.DataFrame, dict)):
            data_generator = [data_generator]
        for num, df in enumerate(data_generator):
            df, status = self.cast_to_data(df)
            if df is None:
                print(status)
                continue
            stat_computation_time = time.time()
            try:
                features = self.get_features(df)
            except self.bench_exceptions:
                continue
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
            bench = self.benchmark(df, timeout, verbose=verbose)
            result["bench"] = bench
            if bench:
                model_data_path = shrynk_path(
                    "{}_{}.jsonl".format(self.model_type, self.model_name)
                )
                self.model_data.append(result)
                if save:
                    with open(model_data_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                results.append(result)

            feature_ids.add(feature_id)
        ### run benchmarks should return a total overview or something
        # return pd.DataFrame(bench).set_index("kwargs")
        return results

    def show_benchmark(self, data, size, write, read, timeout=300):
        data, status = self.cast_to_data(data)
        if data is None:
            raise ValueError(status)
        bench = self.benchmark(data, timeout)
        bench = pd.DataFrame(bench, columns=["kwargs", "size", "write_time", "read_time"])
        return add_z_to_bench(bench, size, write, read)
