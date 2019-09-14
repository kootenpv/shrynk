import os
import shutil
import just
import requests
import lxml.html
from werkzeug import secure_filename

# s = requests.Session()
# r = s.get("https://vincentarelbundock.github.io/Rdatasets/datasets.html")
# tree = lxml.html.fromstring(r.content)

# os.makedirs(os.path.expanduser("~/rcsvs"))

# skip = 0
# for i, x in enumerate(tree.xpath("//a[text() = ' CSV ']/@href")):
#     if skip > i:
#         continue
#     skip = i
#     url = "https://vincentarelbundock.github.io/Rdatasets/" + x
#     try:
#         txt = s.get(url).text
#     except Exception as e:
#         print("error", e)
#         continue
#     with open(os.path.expanduser("~/rcsvs/" + secure_filename(x)), "w") as f:
#         f.write(txt)
#     print(i)

dfs = []
for x in just.glob("/home/pascal/rcsvs/*"):
    try:
        dfs.append(pd.read_csv(x))
    except Exception as e:
        pass

from shrynk.pandas import save, infer, PandasCompressor

# pdc = PandasCompressor("default")
# pdc.run_benchmarks(dfs)

# original size 130M of all .csvs bundled in R packages (blind test-set)
# optimize=write_time        = 113M in 47.7s
# optimize=size              = 21M in 6m29s
# zip each file in folder    = 30M in 5.8s

# run_benchmark & retrain (I THINK IT WENT WRONG)

# optimize=write_time = 104M in 49.5s

# I SHOULD COPY PACKAGE DATA!!!

# optimize=write_time = 103 in 44.3s
# optimize=size       = 21M in 5m12s

# n_estimators = 200
# optimize=write_time = 103M in 58.7s
# optimize=size       = 21M in 5m12s
# BALANCED
# optimize=(3,1,1)    = 31M in 3m45s
# optimize=(1,2,0)    = 36M in 1m34s
# optimize=(1,1,0)    = 34M in 1m8s
# optimize=(1,1,0)    = 34M in 59.1s
# optimize=(1,2,0)    = 36M in 59.5s
# optimize=(0,1,0)    = 55M in 59s
# optimize=(1,1,0) rs = 29M in 1m12s
# optimize=(1,1,0) mm = 35M in 59.3
# optimize=(1,1,0) ma = 35M in 1m
# UNBALANCED
# optimize=(1,1,0) rs = 29M in 1m
# optimize=(1,1,0) z = 29M in 1m
# UNBALANCED CAT
# optimize=(1,1,0) z  = 35M in 51.6s
# optimize=(2,1,0) rs = 25M in 52.7s
# BALANCED CAT
# optimize=(1,1,0) rs = 30M in 49.2s
# optimize=(1,1,0) rs = 31M in 48.3s
# optimize=(0,1,0) rs = 107M in 47.3s
# optimize=(2,1,0) rs = 28M in 51.5s
# optimize=(2,1,0) z = 30M in 48.9s

from catboost import CatBoostClassifier

CatBoostClassifier._get_param_names = lambda: {}

pdc = PandasCompressor("default", clf=CatBoostClassifier, n_estimators=200)
pdc.train_model(2,1,0, scaler="robust_scale", balanced=False)

# for i, df in enumerate(dfs):
#     ddd = pd.DataFrame([pdc.get_features(pd.DataFrame({"a": [1] * 1000}))]).fillna(-100)
#     res = defaultdict(float)
#     res.update([(pdc.arg_lookup[i], x) for i, x in enumerate(pdc.clf.predict_proba()[0])])
#     res.update([(pdc2.arg_lookup[i], x) for i, x in enumerate(pdc2.clf.predict_proba()[0])])
#     best = max(res.items(), key=lambda x: x[1])

if True:
    try:
        os.makedirs("/tmp/rcsvshrink")
    except:
        shutil.rmtree("/tmp/rcsvshrink")
        os.makedirs("/tmp/rcsvshrink")

#

%time for i, df in enumerate(dfs): pdc.save(df, "/tmp/rcsvshrink/{}".format(i))
