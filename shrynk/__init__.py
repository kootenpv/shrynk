""" shrynk - Using Machine Learning as Meta Decision Maker for Compression. """

__project__ = "shrynk"
__version__ = "0.2.21"

import pandas as pd
from shrynk.classes.pandas_ import PandasCompressor
from shrynk.classes.json_ import JsonCompressor
from shrynk.classes.bytes import BytesCompressor

trained_models = {}


def get_compressor(model_type, obj, file_path):
    if isinstance(obj, type(b"")):
        return BytesCompressor
    if isinstance(obj, pd.DataFrame):
        return PandasCompressor
    if (
        "json" in model_type
        or ("json" in file_path and "jsonl" not in file_path)
        or isinstance(obj, (list, dict))
    ):
        return JsonCompressor
    elif "csv" in model_type or "parquet" in file_path or isinstance(obj, pd.DataFrame):
        return PandasCompressor
    elif "bytes" in model_type or isinstance(obj, (type(b""), str)):
        return BytesCompressor
    raise ValueError("No idea what compressor to use")


def _get_model(compressor, model_name, size_write_read):
    key = (model_name, compressor.__name__) + size_write_read
    if key not in trained_models:
        comp = compressor(model_name)
        comp.train_model(*size_write_read)
        trained_models[key] = comp
    return trained_models[key]


def save(
    data,
    fname_prefix,
    inferred_kwargs=None,
    size=3,
    write=1,
    read=1,
    model_type="infer",
    model_name="default",
    allow_overwrite=False,
    **save_kwargs
):
    compressor = get_compressor(model_type, data, fname_prefix)
    comp = _get_model(compressor, model_name, (size, write, read))
    return comp.save(
        data, fname_prefix, inferred_kwargs, allow_overwrite=allow_overwrite, **save_kwargs
    )


def load(file_path, inferred_kwargs=None, **load_kwargs):
    compressor = get_compressor("", "", file_path)
    return compressor.load(file_path, inferred_kwargs, **load_kwargs)


def infer(data, size=3, write=1, read=1, model_type="infer", model_name="default"):
    compressor = get_compressor(model_type, data, "")
    comp = _get_model(compressor, model_name, (size, write, read))
    return comp.infer(data)


def show_benchmark(
    data_or_fname, size=3, write=1, read=1, timeout=300, model_type="infer", model_name="default"
):
    compressor = get_compressor(model_type, data_or_fname, data_or_fname)
    comp = compressor(model_name)
    return comp.show_benchmark(data_or_fname, size, write, read, timeout)


def run_benchmarks(data_or_fname, model_type="infer", model_name="default", save=True):
    compressor = get_compressor(model_type, data_or_fname, data_or_fname)
    comp = compressor(model_name)
    bench = comp.run_benchmarks(data_or_fname, verbose=False, save=save)
    return bench
