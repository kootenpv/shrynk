#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import just
import pandas as pd
from shrynk import save, load, infer, PandasCompressor, show_benchmark


def test_infer():
    infer(pd.DataFrame({"a": [1, 2, 3]}))


def test_infer_path():
    expected = {"engine": "pyarrow", "compression": "brotli"}
    assert PandasCompressor.infer_from_path("bla.pyarrow.brotli") == expected


def test_benchmark_and_train():
    pdc = PandasCompressor("default", n_estimators=10)
    pdc.run_benchmarks([pd.DataFrame({"a": [1, 2, 3, 4]})], ignore_seen=False, save=False)
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


def test_json():
    path = save({"a": [1, 2, 3]}, "test.json")
    try:
        res = load(path)
    finally:
        os.remove(path)


def test_json_gz():
    path = save({"a": [1, 2, 3]}, "test.json", {"compression": "gzip"})
    assert "json" in path
    try:
        res = load(path)
    finally:
        os.remove(path)


def test_bytes():
    path = save(b"asdfasdf", "test", allow_overwrite=True)
    try:
        res = load(path)
    finally:
        os.remove(path)


def test_bytes_png():
    images = just.ls("~/*.png")
    if images:
        path = save(images[0], images[0])
        try:
            res = load(path)
        finally:
            os.remove(path)


def test_bytes_gz():
    path = save(b"asdfasdf", "test", {"compression": "gzip"}, allow_overwrite=True)
    assert "gz" in path
    try:
        res = load(path)
    finally:
        os.remove(path)


def test_show_benchmark_json():
    show_benchmark({"a": [1, 2, 3]}, 0, 1, 1)


def test_show_benchmark_pandas():
    show_benchmark(pd.DataFrame({"a": [1, 2, 3]}), 0, 1, 1)


def test_show_benchmark_bytes():
    show_benchmark(b"asfasfd", 0, 1, 1)
