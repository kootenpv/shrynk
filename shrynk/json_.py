import json
from collections import Counter
import pandas as pd
import just
from shrynk.compressor import BaseCompressor
from shrynk.predictor import Predictor


class JsonCompressor(Predictor, BaseCompressor):
    bench_exceptions = (json.JSONDecodeError, RecursionError)

    compression_options = [
        {"compression": x} for x in ["gz", "bz2", "xz", "zstd"] if x in just.EXT_TO_COMPRESSION
    ] + [{"compression": None}]

    bench_base_path = "jcomp.json"
    model_type = "json"
    # [

    #     {"engine": "csv", "compression": None},
    #     {"engine": "csv", "compression": "gzip"},
    #     {"engine": "csv", "compression": "bz2"},
    #     {"engine": "csv", "compression": "xz"},
    #     {"engine": "csv", "compression": "zip"},
    #     # pyarrow # {‘NONE’, ‘SNAPPY’, ‘GZIP’,  ‘BROTLI’, ‘LZ4’, ‘ZSTD’}
    #     {"engine": "pyarrow", "compression": None},
    #     {"engine": "pyarrow", "compression": "snappy"},
    #     {"engine": "pyarrow", "compression": "gzip"},
    #     {"engine": "pyarrow", "compression": "brotli"},
    #     {"engine": "fastparquet", "compression": "GZIP"},
    #     {"engine": "fastparquet", "compression": "UNCOMPRESSED"},
    #     {"engine": "fastparquet", "compression": "BROTLI"},
    #     # {"engine": "fastparquet", "compression": "LZ4"},
    #     # C
    #     # {"engine": "fastparquet", "compression": "LZO"},
    #     # # # # # # ("fastparquet", "ZSTANDARD"),
    #     # fastparquet can do per column
    #     # pip install fastparquet[brotli]
    #     # pip install fastparquet[lz4]
    #     # pip install fastparquet[lzo]
    #     # pip install fastparquet[zstandard]
    #     # ("fastparquet", {str(x): "BROTLI" if x % 2 == 1 else "GZIP" for x in range(5)})
    # ]

    @classmethod
    def infer_from_path(cls, file_path):
        ending = file_path.split(".")[-1]
        if ending not in just.EXT_TO_COMPRESSION:
            raise ValueError("Do not know how to read this")
        return {"compression": ending}

    def _save(self, obj, file_path_prefix, allow_overwrite=False, compression=None, **save_kwargs):
        path = file_path_prefix
        if not path.endswith(".json"):
            path += ".json"
        if compression is not None:
            path += "." + compression
        if not allow_overwrite and just.exists(path):
            raise ValueError("Path exists, cannot save {!r}".format(path))
        try:
            just.write(obj, path)
        except AttributeError as e:
            print(e)
            raise ValueError("Ensure that file_path_prefix ends with .json")
        return path

    @classmethod
    def load(cls, file_path, inferred_kwargs=None, **load_kwargs):
        return just.read(file_path, **load_kwargs)

    def measure(self, obj, max_level=10, max_consider_list=5, level=0, collector=None):
        if collector is None:
            collector = Counter()
        if level == max_level:
            return collector
        if isinstance(obj, dict):
            collector[level] += len(obj)
            for v in obj.values():
                self.measure(v, max_level, max_consider_list, level + 1, collector)
        elif isinstance(obj, list):
            collector[level] += len(obj)
            for x in obj[:max_consider_list]:
                self.measure(x, max_level, max_consider_list, level + 1, collector)
        else:
            collector[level] += 1
        return collector

    def get_features(self, obj, max_level=10):
        scores = self.measure(obj, max_level=max_level)
        level_scores = []
        for i in range(max_level):
            last = 1 if not level_scores else level_scores[-1]
            level_scores.append(last * max(1, scores[i]))
        return dict(enumerate(level_scores))
