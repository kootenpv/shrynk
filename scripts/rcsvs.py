import os
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

pdc = PandasCompressor("default")
pdc.run_benchmarks(dfs)

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

pdc = PandasCompressor("default")
pdc.train_model("write_time")

pdc2 = PandasCompressor("default", n_estimators=200)
pdc2.train_model("size")


for i, df in enumerate(dfs):
    ddd = pd.DataFrame([pdc.get_features(pd.DataFrame({"a": [1] * 1000}))]).fillna(-100)
    res = defaultdict(float)
    res.update([(pdc.arg_lookup[i], x) for i, x in enumerate(pdc.clf.predict_proba()[0])])
    res.update([(pdc2.arg_lookup[i], x) for i, x in enumerate(pdc2.clf.predict_proba()[0])])
    best = max(res.items(), key=lambda x: x[1])

# %time for i, df in enumerate(dfs): pdc.save(df, "/home/pascal/egoroot/shrynk/rcsvshrink/{}".format(i))
