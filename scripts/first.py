import random
import json
import time
import os
import just
import numpy as np

import tqdm

from features import get_features
from compress import COMPRESSIONS

import just
from itertools import islice
import pandas as pd

g = (x for x in just.glob("/root/data/data/cmc/201*"))

while g:
    paths = list(islice(g, 500))
    data = pd.concat([pd.read_csv(x) for x in paths])
    data.to_parquet(
        paths[0].split("/")[-1].split(".")[0] + ".parquet", engine="pyarrow", compression="brotli"
    )
    for x in paths:
        os.remove(x)
    if len(elements) < 500:
        break

# data = pd.concat([pd.read_csv(x) for x in just.glob("/root/data/data/cmc/20181210*")[:100]])
# for engine, compression in [
#     ("csv", None),
#     ("csv", "gz"),
#     ("csv", "bz2"),
#     ("csv", "xz"),
#     ("csv", "zip"),
#     ("pyarrow", None),
#     ("pyarrow", "snappy"),
#     ("pyarrow", "gzip"),
#     ("pyarrow", "brotli"),
#     ("fastparquet", "GZIP"),
#     ("fastparquet", "UNCOMPRESSED"),
# ]:
#     size, write_time, read_time = time_compression(data, engine, compression)
#     print(engine, compression, size, write_time, read_time)

# 500 gz files (180MB)
#  method                    size(b)       write time      read time
# fastparquet UNCOMPRESSED 140956832 1.2034943103790283 1.481581687927246
# csv None                 141239981 3.090662956237793 27.528740882873535
# csv zip                   62179874 4.2416582107543945 38.110820293426514
# csv gz                    61871349 5.149527072906494 47.88650560379028
# pyarrow None              47558541 0.6397290229797363 1.1741197109222412
# fastparquet GZIP          43452083 4.606407165527344 11.498911142349243
# csv bz2                   43416518 17.830265045166016 45.81295394897461
# pyarrow snappy            40789362 0.5927798748016357 1.1275267601013184
# pyarrow gzip              34002496 0.9507162570953369 4.055440664291382
# csv xz                    29750900 6.5890185832977295 242.52700304985046
# pyarrow brotli            29452478 0.8912572860717773 6.203507900238037

# 100 gz
#  method                    size(b)       write time      read time
# fastparquet UNCOMPRESSED 28196091 0.24243664741516113 0.3680732250213623
# csv None                 28234482 0.6111953258514404 5.691488742828369
# csv zip                  12434684 0.8487372398376465 7.760023355484009
# csv gz                   12372919 nan 9.925037860870361
# pyarrow None             10498508 0.13428807258605957 0.3446931838989258
# fastparquet GZIP          8721683 0.41844701766967773 2.0790317058563232
# csv bz2                   8683147 3.5419511795043945 9.14890170097351
# pyarrow snappy            9071102 0.09588813781738281 0.2966160774230957
# pyarrow gzip              7754594 0.1672370433807373 0.9379496574401855
# csv xz                    6113612 1.347106695175171 46.70905303955078
# pyarrow brotli            7128701 0.1787869930267334 1.1302130222320557

# 1 gz
#  method                    size(b)       write time      read time
# fastparquet UNCOMPRESSED 270055 0.010703325271606445 0.031023502349853516
# csv None                 281095 0.017310142517089844 0.05772900581359863
# csv zip                  124968 0.016399383544921875 0.07457542419433594
# csv gz                   124254 nan 0.09394264221191406
# pyarrow None             262880 0.007834434509277344 0.007164955139160156
# fastparquet GZIP         125170 0.015429496765136719 0.05287432670593262
# csv bz2                  104706 0.03296542167663574 0.09511423110961914
# pyarrow snappy           203199 0.007740020751953125 0.007920980453491211
# pyarrow gzip             157221 0.008668899536132812 0.024445772171020508
# csv xz                   105820 0.027971506118774414 0.23724055290222168
# pyarrow brotli           150970 0.009110212326049805 0.030408143997192383


def write_error(fname):
    with open("/home/pascal/csvres.jsonl", "a") as f:
        f.write(json.dumps({"meta": {"fname": fname}, "error": True}) + "\n")


if __name__ == "__main__":
    tot = 0
    results = []
    done = set()
    with open("/home/pascal/csvres.jsonl") as f:
        for line in f:
            done.add(json.loads(line)["meta"]["fname"])

    lines = just.read("~/csvlist.txt").split("\n")
    for line in tqdm.tqdm(random.sample(lines, len(lines))):
        if line in done:
            continue
        try:
            mb = os.path.getsize(line) / 1000 / 1000
        except FileNotFoundError:
            done.add(line)
            write_error(line)
            continue
        # tot += mb
        # if mb > 1:
        #     print(mb, line)
        df, meta, X = get_meta(line)
        if X is not None:
            # print(mb, line)
            print()
        done.add(line)
        for res in results:
            with open("/home/pascal/csvres.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
        results = []


# a="""#  method                    size(b)       write time      read time
# # fastparquet UNCOMPRESSED 28196091 0.24243664741516113 0.3680732250213623
# # csv None                 28234482 0.6111953258514404 5.691488742828369
# # csv zip                  12434684 0.8487372398376465 7.760023355484009
# # csv gz                   12372919 0.5 9.925037860870361
# # pyarrow None             10498508 0.13428807258605957 0.3446931838989258
# # fastparquet GZIP          8721683 0.41844701766967773 2.0790317058563232
# # csv bz2                   8683147 3.5419511795043945 9.14890170097351
# # pyarrow snappy            9071102 0.09588813781738281 0.2966160774230957
# # pyarrow gzip              7754594 0.1672370433807373 0.9379496574401855
# # csv xz                    6113612 1.347106695175171 46.70905303955078
# # pyarrow brotli            7128701 0.1787869930267334 1.1302130222320557"""

# df = pd.DataFrame([((" ".join(x.split()[1:3]),) + tuple([float(x) for x in x.split()[3:]])) for x in a.split("\n")[1:]])
# df.columns = ["name", "comp", "write", "read"]
# df["sorted"] = (4 * (1.01 - (df.comp / df.comp.max()))) / (3 * df.write) / df.read
# df.sort_values("sorted")
