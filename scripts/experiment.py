import json
import just
import random
import pandas as pd
import numpy as np

from shrynk.pandas import PandasCompressor

pdc = PandasCompressor("default", n_estimators=100)
for i in range(5):
    pdc.train_model((3, 2, 1), n_validations=1)


pdc.run_benchmarks([pd.DataFrame({"a": [1]})], save=False)

# from shrynk.pandas import PandasCompressor
from sklearn.preprocessing import scale

with open("/home/pascal/shrynk_new.jsonl") as f:
    data = [json.loads(x) for x in f.read().split("\n") if x]

upsample = False
targets = ["size", "write_time", "read_time"]
size_write_read = np.array((0, 1, 0))
features = []
y = []
feature_ids = []
for x in data:
    z = (
        (scale([[y[t] for t in targets] for y in x["bench"]]) * size_write_read)
        .sum(axis=1)
        .argmin()
    )
    features.append(x["features"])
    y.append(x["bench"][z]["kwargs"])
    feature_ids.append(x["feature_id"])

features = pd.DataFrame(features)
features["y"] = y
features["feature_id"] = feature_ids

features = features.sort_values("y")
features["train"] = features.index % 2 == 0

if upsample:
    largest_n = features["y"].astype(str).value_counts().max()
    features = (
        features.groupby("y")
        .apply(lambda x: x.sample(largest_n, replace=True))
        .reset_index(drop=True)
    )

X_train = features.query("train").drop(["y", "train", "feature_id"], axis=1)
y_train = features.query("train")["y"]
X_test = features.query("~train").drop(["y", "train", "feature_id"], axis=1)
y_test = features.query("~train")["y"]

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train.fillna(-100), y_train)
print("acc", np.mean(clf.predict(X_test.fillna(-100)) == y_test))

preds = clf.predict(X_test.fillna(-100))

test = features.query("~train")
test["preds"] = preds

lookup = {x["feature_id"]: x["bench"] for x in data}
scores = []
for fid, group in test.groupby("feature_id"):
    # bla = pd.DataFrame(lookup[fid])
    # if group.shape[0] != 1:
    #    "a"+1
    # pred = group.preds.iloc[0]
    # bla["z"] = (scale([[y[t] for t in targets] for y in lookup[fid]]) * weights).sum(axis=1)
    # bla = bla.sort_values("z")
    bench = pd.DataFrame(lookup[fid]).drop_duplicates("kwargs")
    bench["z"] = (scale([[y[t] for t in targets] for y in lookup[fid]]) * weights).sum(axis=1)
    try:
        pred = bench[bench.kwargs == group.iloc[0].preds].z.iloc[0]
        best = bench[bench.kwargs == group.iloc[0].y].z.iloc[0]
        scores.append(best - pred)
        if best > pred:
            "a" + 1
    except IndexError:
        continue


pdc = PandasCompressor("default", n_estimators=200, max_depth=5)
pdc.train_model("auto", auto_weights=(5, 4, 3))

pdc2 = PandasCompressor("default", n_estimators=200, max_depth=5)
pdc2.train_model("size")

# auto_weights=(size, write_time, read_time)
# higher means it has more influence
# pdc.train_model("auto", auto_weights=(5, 4, 3))
#                  winners                                  times
# {'engine': 'csv', 'compression': 'bz2'}                     144
# {'engine': 'csv', 'compression': 'zip'}                      50
# {'engine': 'fastparquet', 'compression': 'GZIP'}             42
# {'engine': 'csv', 'compression': 'gzip'}                     26
# {'engine': 'fastparquet', 'compression': 'LZO'}              12
# {'engine': 'csv', 'compression': None}                        6
# {'engine': 'pyarrow', 'compression': 'brotli'}                4
# {'engine': 'pyarrow', 'compression': 'snappy'}                3
# {'engine': 'fastparquet', 'compression': 'LZ4'}               3
# {'engine': 'fastparquet', 'compression': 'UNCOMPRESSED'}      3
# {'engine': 'pyarrow', 'compression': 'gzip'}                  2

# pdc.train_model("auto", auto_weights=(1,3,1))
#                  winners                                  times
# {'engine': 'csv', 'compression': 'zip'}                     104
# {'engine': 'csv', 'compression': 'bz2'}                      53
# {'engine': 'csv', 'compression': 'gzip'}                     40
# {'engine': 'fastparquet', 'compression': 'GZIP'}             29
# {'engine': 'fastparquet', 'compression': 'LZO'}              22
# {'engine': 'pyarrow', 'compression': 'snappy'}               21
# {'engine': 'pyarrow', 'compression': 'gzip'}                  8
# {'engine': 'csv', 'compression': None}                        7
# {'engine': 'pyarrow', 'compression': 'brotli'}                6
# {'engine': 'fastparquet', 'compression': 'UNCOMPRESSED'}      3
# {'engine': 'pyarrow', 'compression': None}                    1
# {'engine': 'fastparquet', 'compression': 'LZ4'}               1

# pdc.train_model("auto", auto_weights=(1,3,3))
# ## -- End pasted text --
# from package
# {'engine': 'csv', 'compression': 'zip'}                     96
# {'engine': 'csv', 'compression': 'gzip'}                    51
# {'engine': 'pyarrow', 'compression': 'snappy'}              36
# {'engine': 'fastparquet', 'compression': 'LZO'}             33
# {'engine': 'csv', 'compression': 'bz2'}                     27
# {'engine': 'fastparquet', 'compression': 'GZIP'}            16
# {'engine': 'csv', 'compression': None}                      15
# {'engine': 'pyarrow', 'compression': 'gzip'}                 7
# {'engine': 'pyarrow', 'compression': 'brotli'}               6
# {'engine': 'pyarrow', 'compression': None}                   4
# {'engine': 'fastparquet', 'compression': 'LZ4'}              2
# {'engine': 'fastparquet', 'compression': 'UNCOMPRESSED'}     2

X_train, y_train, X_test, y_test = pdc.train_model("size", n_validations=5)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

clf = RandomForestClassifier(n_estimators=200, max_depth=5)
np.mean(clf.fit(X_train, y_train).predict(X_test) == y_test)

import joblib

with open("a", "wb") as f:
    joblib.dump(clf, f)

g  # 10 estimators, 5 vals
# (0.837389770723104,
#  defaultdict(dict,
#              {"{'engine': 'csv', 'compression': 'bz2'}": 43.911446311185905,
#               "{'engine': 'csv', 'compression': 'xz'}": 55.550280387538045,
#               "{'engine': 'fastparquet', 'compression': 'BROTLI'}": 607.0942320226405,
#               "{'engine': 'pyarrow', 'compression': 'brotli'}": 1258.5016274768464,
#               "{'engine': 'csv', 'compression': 'None'}": 312.8534126262998,
#               "{'engine': 'fastparquet', 'compression': 'GZIP'}": 689.0448268241357}))
# CPU times: user 4min 27s, sys: 158 ms, total: 4min 27s

# 100 estimators, 5 vals
# (0.7529411764705882,
#  defaultdict(dict,
#              {"{'engine': 'csv', 'compression': 'bz2'}": 37.78959825062472,
#               "{'engine': 'csv', 'compression': 'xz'}": 49.13666602390679,
#               "{'engine': 'pyarrow', 'compression': 'brotli'}": 1181.5739669734976,
#               "{'engine': 'fastparquet', 'compression': 'BROTLI'}": 571.8744450633159,
#               "{'engine': 'fastparquet', 'compression': 'GZIP'}": 660.3157093830315,
#               "{'engine': 'csv', 'compression': 'None'}": 333.4312455953604}))
# CPU times: user 4min 52s, sys: 85.8 ms, total: 4min 52s

# 200 estimators, 5 vals
# (0.5036168620603642,
#  defaultdict(dict,
#              {"{'engine': 'csv', 'compression': 'bz2'}": 51.98367826299148,
#               "{'engine': 'fastparquet', 'compression': 'BROTLI'}": 675.8711840248326,
#               "{'engine': 'csv', 'compression': 'xz'}": 65.18381894387288,
#               "{'engine': 'pyarrow', 'compression': 'brotli'}": 1417.8895178446014,
#               "{'engine': 'csv', 'compression': 'None'}": 310.13647332174526}))
# CPU times: user 4min 20s, sys: 144 ms, total: 4min 20s
# Wall time: 4min 23s


df = pd.DataFrame(np.random.random((10000, int(random.random() * 500))))

pdc.run_benchmarks([df], save=True)

df = pd.DataFrame({"a": [1]})
inferred = pdc.infer(df)
inferred
pdc.save(df, "test")
pdc.save(df, "~/test")
path = pdc.save(df, "~/test")
pdc.load(path)

# pdc.bench_exceptions = ()
# data_gen = (pd.DataFrame(np.random.random((1000, int(random.random() * 500)))) for i in range(2))

pdc.run_benchmarks(just.iread("~/csvlist.txt"))

pdc.train_model("size")

feats = json.loads(
    '{"num_obs": 1000, "num_cols": 289, "num_float_vars": 289, "num_str_vars": 0, "percent_float": 1.0, "percent_str": 0.0, "str_missing_proportion": NaN, "float_missing_proportion": NaN, "cardinality_quantile_proportion_25": 1.0, "cardinality_quantile_proportion_50": 1.0, "cardinality_quantile_proportion_75": 1.0, "float_equal_0_proportion": 0.0, "str_len_quantile_25": NaN, "str_len_quantile_50": NaN, "str_len_quantile_75": NaN}'
)

pdc.predict(pd.DataFrame(np.random.random((10000, int(random.random() * 500000)))))

# pdc = PandasCompressor("test1")

# d = pdc.get_features(pd.read_csv("/home/pascal/Downloads/results-20190622-143623.csv"))

# fuck = just.glob("/home/pascal/egoroot/tradex/data/cmc/*.csv.gz") * 30

# df = pd.concat([pd.read_csv(x) for x in fuck])

# for args in pdc.compression_options:
#     size, write_time, read_time = pdc.benchmark(df, *args)
#     print(*args, size, write_time, read_time)


# for name, group in results.groupby("y"):
#     res = []
#     cnames = []
#     for cname, clf in clfs.items():
#         res.append(clf.predict(group[1::2][X_cols]))
#         cnames.append(cname)
#     cnames = np.array(cnames)
#     Z = np.array(res).T
#     fuck = []
#     for x, y in zip(Z, np.argmin(Z, axis=1)):
#         fuck.append(x[y])
#     fuck = np.array(fuck)
#     # print("class correct", np.mean(cnames[np.argmin(Z, axis=1)] == group["bestclass"][1::2]))
#     print(fuck.shape, group["bestvalue"][1::2].shape)
#     print("percentage", fuck.sum() / group["bestvalue"][1::2].sum())

# maping = list(results["y"].unique())
# results["y_num"] = [maping.index(x) for x in results["y"]]
# X = train[list(X_cols) + ["y_num"]]


# def mape_metric(preds, real):
#     return "mape", np.sum(np.abs(1 - (preds / real)))


# y = train["size"]
# clf2 = RandomForestRegressor(n_estimators=1000, feval=mape_metric)
# clf2.fit(X[::2], np.log(y[::2]))

# from sklearn.preprocessing import scale
# weights = [5, 4, 3]
# df = pd.DataFrame({"size": [1,2,3,4], "write": [1,2,3,4], "read": [1,2,3,4]}, index=list("abcd"), columns=["size", "write", "read"])
# print(df)
# df[:] = scale(df) * np.array([weights])
# df.sum(axis=1).idxmin()
