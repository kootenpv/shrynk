#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from shrynk.pandas import save, load, infer, PandasCompressor


def test_infer():
    infer(pd.DataFrame({"a": [1, 2, 3]}))


def test_infer_path():
    expected = {"engine": "pyarrow", "compression": "brotli"}
    assert PandasCompressor.infer_from_path("bla.pyarrow.brotli") == expected


def test_benchmark_and_train():
    pdc = PandasCompressor("default", n_estimators=10)
    pdc.run_benchmarks([pd.DataFrame({"a": [1, 2, 3, 4]})], save=False, ignore_seen=False)
    pdc.train_model(4, 1, 1)


def test_pandas():
    path = save(pd.DataFrame({"a": [1]}), "test")
    try:
        res = load(path)
    finally:
        os.remove(path)


def test_pyarrow_pandas():
    path = save(pd.DataFrame({"a": [1]}), "test", {"engine": "pyarrow", "compression": None})
    try:
        res = load(path)
    finally:
        os.remove(path)


def test_pyarrow_brotli_pandas():
    path = save(pd.DataFrame({"a": [1]}), "test", {"engine": "pyarrow", "compression": "brotli"})
    try:
        res = load(path)
    finally:
        os.remove(path)
