<p align="center">
  <img src="./Shrynk.png" width="400rem"/>
</p>

[![PyPI](https://img.shields.io/pypi/pyversions/shrynk.svg?style=flat-square&logo=python)](https://pypi.python.org/pypi/shrynk/)
[![PyPI](https://img.shields.io/pypi/v/shrynk.svg?style=flat-square&logo=pypi)](https://pypi.python.org/pypi/shrynk/)
[![HitCount](http://hits.dwyl.io/kootenpv/shrynk.svg)](http://hits.dwyl.io/kootenpv/shrynk)

Try it live at https://shrynk.ai

### Usage

Installation:

    pip install shrynk

Then in Python:

```python
from shrynk.pandas import save, load
file_path = save(my_df, "mypath")
# e.g. mypath.csv.bz2
loaded_df = load(file_path)
```

### Add your own data

If you want more control you can do the following:

```python
import pandas as pd
from shrynk.pandas import PandasCompressor

df = pd.DataFrame({"a": [1, 2, 3]})

pdc = PandasCompressor("default")
pdc.run_benchmarks([df], save=False) # adds data to the default

pdc.train_model(size=3, write=1, read=1)

pdc.infer(df)
```
