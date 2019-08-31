import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

results = []
with open("/home/pascal/csvres.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

for x in results:
    if "error" not in x and "read_time" in x["meta"]:
        x["meta"]["uncompressed_read_time"] = x["meta"].pop("read_time")


results = pd.DataFrame(results)
results = results[results["X"].notnull()]
X_cols = pd.io.json.json_normalize(results["X"]).columns

results = pd.concat(
    (
        results,
        pd.io.json.json_normalize(results["X"]),
        pd.io.json.json_normalize(results["result"]),
        pd.io.json.json_normalize(results["meta"]),
    ),
    axis=1,
)

results = results[results["X"].notnull()]
results.drop("error", axis=1, inplace=True)

results.drop("X", axis=1, inplace=True)
results.drop("meta", axis=1, inplace=True)
results.drop("result", axis=1, inplace=True)
results = results[results.num_obs > 0]

results["size (MB)"] = results["size"] / 1000 / 1000

ok_fnames = np.array(
    [name for name, group in results.groupby("fname") if (group["size"] > 0).sum() == 11]
)
results = results[results["fname"].isin(ok_fnames)]

results["y"] = results["engine"] + " " + results["compression"].astype(str)

target = "size"


def myfunc(group):
    best = group.loc[group[target].idxmin()]
    group["bestclass"] = best["y"]
    group["bestvalue"] = best[target]
    return group


results = results.groupby("fname").apply(myfunc)

class_weights = 1 / (
    results["bestclass"].value_counts() / results["bestclass"].value_counts().sum()
)
results["sample_weight"] = [class_weights.get(x, 0) for x in results["y"]]

best = results.loc[results.groupby("fname")[target].idxmin()]
X = best[X_cols]
y = best["y"]


clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X[::2].fillna(-100), y[::2])
# clf.fit(X[::2], y[::2], sample_weight=best["sample_weight"][::2])
preds = clf.predict(X[1::2].fillna(-100))
weights = best["sample_weight"][1::2]
print("acc", np.mean(preds == y[1::2]))
print("f1 weighted micro", f1_score(preds, y[1::2], average="micro", sample_weight=weights))
print("f1 weighted macro", f1_score(preds, y[1::2], average="macro", sample_weight=weights))
print("f1 unweighted micro", f1_score(preds, y[1::2], average="micro"))
print("f1 unweighted macro", f1_score(preds, y[1::2], average="macro"))

# clf2 = RandomForestClassifier(1000, class_weight="balanced")
# clf2.fit(XX[::2], y[::2])
# print(np.mean(clf2.predict(XX[1::2]) == y[1::2]))

bla = []
for strategy in group.y.unique():
    print("strategy", strategy)
    total_loss = []
    it = -1
    for name, group in results.groupby("fname"):
        it += 1
        if it % 2 == 0:
            continue
        res = group[group.y == strategy]
        if res.empty:
            continue
        tot = res['size'].iloc[0] / res['bestvalue'].iloc[0]
        total_loss.append(tot)
    bla.append((strategy, np.mean(total_loss)))

X_train = []
y_train = []
it = -1
for name, group in results.groupby("fname"):
    it += 1
    if it % 2 != 0:
        continue
    X_train.append(group.iloc[0][X_cols])
    y_train.append(group.iloc[0].bestclass)
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

total_loss = []
it = -1
for name, group in results.groupby("fname"):
    it += 1
    if it % 2 == 0:
        continue
    pred = clf.predict(group.iloc[:1][X_cols].fillna(-100))[0]
    res = group[group.y == pred]
    if res.empty:
        continue
    tot = res['size'].iloc[0] / res['bestvalue'].iloc[0]
    total_loss.append(tot)

# def applypredvalue(group):
#     group["predvalue"] = group[group["y"] == group["pred"]][target]
#     return group


# results["pred"] = clf.predict(results[X_cols])

# results = results.groupby("fname").apply(applypredvalue)

# test_fnames = np.array(best["fname"][1::2])
# vali_rows = results[results["fname"].isin(test_fnames)]

# pred_records = vali_rows[vali_rows["predvalue"].notnull()]
# print(
#     "Strategy: prediction, relative", np.mean(pred_records["predvalue"] / pred_records["bestvalue"])
# )

# print("Strategy: single method, relative")
# print(vali_rows.groupby("y").apply(lambda g: (g[target] / g["bestvalue"]).mean()))

# print(
#     "Strategy: prediction, cumulative total",
#     pred_records["predvalue"].sum() / pred_records["bestvalue"].sum(),
# )
# print("Strategy: single method, relative")
# print(vali_rows.groupby("y").apply(lambda g: g[target].sum() / g["bestvalue"].sum()))

# confusion_matrix(preds, np.array(y[1::2]))

# # ## #

# In [148]: np.round(results[["num_obs", "num_cols", "size (MB)", "read_time", "write_time"]].describe(), 4)
# Out[148]:
#              num_obs    num_cols   size (MB)   read_time  write_time
# count     19041.0000  19041.0000  19041.0000  19041.0000  18873.0000
# mean      42737.6528     45.4044      1.7838      0.9239      0.5585
# std      410534.6577    408.1824     17.7581     15.9603     26.7128
# min           1.0000      2.0000      0.0000      0.0004      0.0008
# 25%          84.0000      4.0000      0.0037      0.0017      0.0021
# 50%         252.0000      7.0000      0.0142      0.0051      0.0029
# 75%        1796.0000     14.0000      0.0823      0.0195      0.0070
# max    10182251.0000  10002.0000   1019.0941    986.6062   1868.0859

# # # # WRITE_TIME

# acc 0.4913294797687861
# f1 weighted micro 0.3227472294607821
# f1 weighted macro 0.27958397782501593
# f1 unweighted micro 0.4913294797687861
# f1 unweighted macro 0.2887686688652174

# Strategy: prediction, relative 1.2750379715603046
# Strategy: single method, relative
# y
# csv None                    3.3750786206
# csv bz2                     7.6378961202
# csv gz                      4.0914528187
# csv xz                      5.5207703505
# csv zip                     3.7270183228
# fastparquet GZIP            2.5719561333
# fastparquet UNCOMPRESSED    1.4219452095
# pyarrow None                1.8895763114
# pyarrow brotli              2.0665753259
# pyarrow gzip                3.1323341931
# pyarrow snappy              1.8938357773
# dtype: float64
# Strategy: prediction, cumulative total 2.442810075912227
# Strategy: single method, relative
# y
# csv None                     4.1674076531
# csv bz2                     10.3440510329
# csv gz                       5.0813294286
# csv xz                      12.3705499203
# csv zip                     16.3250404431
# fastparquet GZIP             4.3092469995
# fastparquet UNCOMPRESSED     1.7517606061
# pyarrow None                12.9382708081
# pyarrow brotli               5.1838827037
# pyarrow gzip                 9.3984839456
# pyarrow snappy              12.9524235988
# dtype: float64
# CPU times: user 14.1 s, sys: 9.85 ms, total: 14.2 s
# Wall time: 14.2 s
