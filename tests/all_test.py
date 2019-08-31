#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from shrynk.pandas import save, load


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
