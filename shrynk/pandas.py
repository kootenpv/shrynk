import re
import os
import zipfile

import numpy as np
import pandas as pd
from pyarrow import ArrowTypeError


from shrynk.compressor import BaseCompressor
from shrynk.predictor import Predictor


def safelen(x):
    try:
        return len(x)
    except TypeError:
        return 0


# story:
# in the setup of the compressor we should use the basics options that don't require extras_require
# in the extras we can add the ones we want, which will then be imported
# - default model is always included in the package
# - for extras, it will download the data and it will train a model?
# for models, set the random_seed to 42

from pandas.io.common import _compression_to_extension

_csv_opts = [
    {"engine": "csv", "compression": x} for x in [None] + list(_compression_to_extension.keys())
]

# OPTIONAL: load pyarrow
try:
    import pyarrow
    from pyarrow import ArrowTypeError, ArrowNotImplementedError, ArrowInvalid

    _pyarrow_exceptions = (ArrowTypeError, ArrowNotImplementedError, ArrowInvalid)
    _pyarrow_opts = [
        [
            {"engine": "pyarrow", "compression": y}
            for y in re.split("[', {}]+", x.split(": ")[1])
            if y
        ]
        for x in pyarrow.compress.__doc__.split("\n")
        if "upported types" in x
    ][0]
except ImportError:
    arrow_exceptions = ()
    _pyarrow = []

# OPTIONAL: load fastparquet
try:
    from fastparquet.compression import compressions

    # BROTLI IS BUGGED!
    _fastparquet_opts = [
        {"engine": "fastparquet", "compression": x} for x in compressions.keys() if x != "BROTLI"
    ]
except ImportError:
    _fastparquet_opts = []


class PandasCompressor(Predictor, BaseCompressor):
    bench_exceptions = (
        ValueError,
        pd.errors.ParserError,
        zipfile.BadZipFile,
        UnicodeDecodeError,
        OSError,
        pd.errors.EmptyDataError,
    ) + _pyarrow_exceptions

    compression_options = _fastparquet_opts + _pyarrow_opts + _csv_opts
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
        endings = file_path.split(".")[-2:]
        engine = None
        compression = None
        for x in cls.compression_options:
            if x["engine"] in endings:
                engine = x["engine"]
            if x["compression"] in endings:
                compression = x["compression"]
        if engine is None and "parquet" in endings:
            engine = "auto"
        if engine == "csv" and compression is None and len(endings) > 1:
            compression = "infer"
        return {"engine": engine, "compression": compression}

    def _save(self, df, file_path_prefix, engine=None, compression=None, **save_kwargs):
        if file_path_prefix.endswith(".csv") or file_path_prefix.endswith(".parquet"):
            raise ValueError("Only provide a base path, not an extension, the path wil be returned")
        path = "{}.shrynk.{}.{}".format(file_path_prefix, engine, compression)
        if engine is None or engine == "csv":
            df.to_csv(path, **save_kwargs)
        else:
            df.columns = [str(x) for x in df.columns]
            df.to_parquet(path, engine=engine, compression=compression, **save_kwargs)
        return path

    @classmethod
    def load(cls, file_path, inferred_kwargs=None, **load_kwargs):
        if inferred_kwargs is None:
            inferred_kwargs = cls.infer_from_path(file_path)
        if inferred_kwargs["engine"] is None or inferred_kwargs["engine"] == "csv":
            data = pd.read_csv(
                file_path, compression=inferred_kwargs.get("compression"), **load_kwargs
            )
        else:
            data = pd.read_parquet(file_path, engine=inferred_kwargs["engine"], **load_kwargs)
        return data

    def get_features(self, df):
        num_cols = df.shape[1]
        num_obs = df.shape[0]
        float_cols = [c for c, d in zip(df.columns, df.dtypes) if "float" in str(d)]
        str_cols = [c for c, d in zip(df.columns, df.dtypes) if "object" in str(d)]
        if str_cols and not df[str_cols].empty:
            str_len_quantiles = list(
                df[str_cols].applymap(safelen).quantile([0.25, 0.50, 0.75], axis=0).mean(axis=1)
            )
            str_missing_proportion = (~df[str_cols].notnull()).mean().mean()
        else:
            str_len_quantiles = [0, 0, 0]
            str_missing_proportion = 0
        if float_cols and not df[float_cols].empty:
            float_equal_0 = (df[float_cols] == 0).mean().mean()
            float_missing_proportion = (~df[str_cols].notnull()).mean().mean()
        else:
            float_equal_0 = 0
            float_missing_proportion = 0
        cardinality = df.apply(pd.Series.nunique)
        cardinality_quantile_proportion = cardinality.quantile([0.25, 0.50, 0.75]) / num_obs
        memory_usage = df.memory_usage().sum()
        data = {
            # standard
            "num_obs": num_obs,
            "num_cols": num_cols,
            "num_float_vars": len(float_cols),
            "num_str_vars": len(str_cols),
            "percent_float": len(float_cols) / num_cols,
            "percent_str": len(str_cols) / num_cols,
            "str_missing_proportion": str_missing_proportion,
            "float_missing_proportion": float_missing_proportion,
            "cardinality_quantile_proportion_25": cardinality_quantile_proportion.iloc[0],
            "cardinality_quantile_proportion_50": cardinality_quantile_proportion.iloc[1],
            "cardinality_quantile_proportion_75": cardinality_quantile_proportion.iloc[2],
            # extra cpu
            "float_equal_0_proportion": float_equal_0,
            "str_len_quantile_25": str_len_quantiles[0],
            "str_len_quantile_50": str_len_quantiles[1],
            "str_len_quantile_75": str_len_quantiles[2],
            "memory_usage": memory_usage,
        }
        return data

    def is_valid(self, df):
        if df.shape[1] < 2:
            return None
        if df.empty:
            return None
        return df


# - set of ids
# X = [x["features"] for x in results]
# allowed_kwargs = set(['{"compression": null, "engine": "csv"}'])
# y = [min([y for y in x["bench"] if y["kwargs"] in allowed_kwargs], key=lambda x: x[target])["kwargs"] for x in results]
# -- for quick bench lookup
#

trained_models = {}


def _get_model(model_name, size_write_read):
    key = (model_name,) + size_write_read
    if key not in trained_models:
        pdc = PandasCompressor(model_name)
        pdc.train_model(*size_write_read)
        trained_models[key] = pdc
    return trained_models[key]


def save(
    df,
    fname_prefix,
    inferred_kwargs=None,
    size=3,
    write=1,
    read=1,
    model_name="default",
    **save_kwargs
):
    pdc = _get_model(model_name, (size, write, read))
    return pdc.save(df, fname_prefix, inferred_kwargs, **save_kwargs)


def load(file_path, inferred_kwargs=None, **load_kwargs):
    return PandasCompressor.load(file_path, inferred_kwargs, **load_kwargs)


def infer(df, size=1, write=0, read=0, model_name="default"):
    pdc = _get_model(model_name, (size, write, read))
    return pdc.infer(df)
