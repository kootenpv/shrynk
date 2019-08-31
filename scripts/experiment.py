import json
import just
import random
import pandas as pd
import numpy as np
from shrynk.pandas import PandasCompressor

pdc = PandasCompressor("default", n_estimators=200, max_depth=5)

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
