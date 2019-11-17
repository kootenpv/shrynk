import os
import just
from shrynk.compressor import BaseCompressor
from shrynk.predictor import Predictor


class BytesCompressor(Predictor, BaseCompressor):
    bench_exceptions = (UnicodeDecodeError, PermissionError)

    compression_options = [
        {"compression": x} for x in ["gz", "bz2", "xz", "zstd"] if x in just.EXT_TO_COMPRESSION
    ] + [{"compression": None}]

    bench_base_path = "jcomp.bytes"
    model_type = "bytes"
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

    def _save(
        self, fname_or_obj, file_path_prefix, allow_overwrite=False, compression=None, **save_kwargs
    ):
        if isinstance(fname_or_obj, str):
            fname = os.path.expanduser(fname_or_obj)
            if os.path.isdir(fname):
                raise ValueError("Cannot compress directory")
            elif os.path.isfile(fname):
                obj = just.bytes.read(fname)
            else:
                obj = fname_or_obj
        else:
            obj = fname_or_obj
        path = file_path_prefix
        if compression is not None:
            path += "." + compression
        if not allow_overwrite and just.exists(path):
            raise ValueError("Path exists, cannot save {!r}".format(path))
        if isinstance(obj, str):
            obj = bytes(obj, encoding="utf8")
        just.write(obj, path, unknown_type="bytes")
        return path

    @classmethod
    def load(cls, file_path, inferred_kwargs=None, **load_kwargs):
        # deferred loading till later
        return file_path

    def cast_to_data(self, df):
        if isinstance(df, str) and os.path.isfile(df):
            try:
                df = self.load(df)
            except self.bench_exceptions as e:
                print(e)
                return None, str(e)
        if df is None:
            return None, "df is None"
        return df, "OK"

    def get_features(self, fname_or_obj):
        if isinstance(fname_or_obj, str):
            fname = os.path.expanduser(fname_or_obj)
            if os.path.isdir(fname):
                raise ValueError("Cannot get features for dir?")
            elif os.path.isfile(fname):
                with open(fname, "rb") as f:
                    # first 14 bytes become features
                    features = dict(enumerate(f.read(14)))
                features["file_size"] = os.stat(fname).st_size
                _, file_extension = os.path.splitext(fname)
                if file_extension:
                    features[file_extension] = 1
                features = {str(i): x for i, x in features.items()}
                return features
            else:
                fname_or_obj = bytes(fname_or_obj, encoding="utf8")
        features = dict(enumerate([x for x in fname_or_obj[:14]]))
        features = {str(i): x for i, x in features.items()}
        return features
