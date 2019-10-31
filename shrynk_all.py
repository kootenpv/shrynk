import os
import sys
import random
import warnings

WEIGHTS = tuple([int(x) for x in sys.argv[-3:]])

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.stderr = open(os.devnull, "w")  # silence stderr

import just
import shrynk
from shrynk.pandas import PandasCompressor
from shrynk.json import JsonCompressor

print()
print(
    "===============================================================\n"
    "[shrynk=={}] | User Defined Weights: size={}, write={} read={}".format(
        shrynk.__version__, *WEIGHTS
    )
    + "\n==============================================================="
)
print()
pdc = PandasCompressor("default")
jc = JsonCompressor("json_default")

print("Training RandomForest Model on packaged benchmark results...")
print()
pdc.train_model(*WEIGHTS)
jc.train_model(*WEIGHTS)

sys.stderr = sys.__stderr__  # unsilence stderr

from colorama import Fore


def human_readable_size(size, decimal_places=3):
    for unit in ['b', 'kb', 'mb', 'gb', 'tb']:
        if size < 1000.0:
            break
        size /= 1000.0
    return f"{size:.{decimal_places}f}{unit}"


dir_path = "/home/pascal/shrynk"
print("Processing:", dir_path)

old_total = 0
new_total = 0
if not dir_path.endswith("/"):
    dir_path += "/"

fnames = just.glob(dir_path + "*.json") + just.glob(dir_path + "*.csv")
random.shuffle(fnames)

for x in fnames:
    print(x)
    if x.endswith(".json"):
        shrynk = jc
        tp = "JSON"
    elif x.endswith(".csv"):
        shrynk = pdc
        tp = "CSV"
    old_size = os.path.getsize(x)
    old_total += old_size
    data = shrynk.load(x)
    new_file = shrynk.save(data, x.replace("csv_", "").replace(".csv", "").replace(".json", ""))
    comp = new_file.split(".")[-1]
    if comp == "None":
        new_size = old_size
    else:
        new_size = os.path.getsize(new_file)
    new_total += new_size
    just.remove(new_file)
    improvement = 100 - int(new_size / old_size * 100)
    if improvement > 50:
        improvement = Fore.GREEN + "{}%".format(improvement) + Fore.RESET
    elif improvement > 20:
        improvement = Fore.YELLOW + "{}%".format(improvement) + Fore.RESET
    else:
        improvement = Fore.RED + "{}%".format(improvement) + Fore.RESET
    improvement = improvement.rjust(13)
    print(
        "•",
        tp.rjust(4),
        "->",
        comp.rjust(5),
        "{} (deflated {})".format(human_readable_size(new_size, 0), improvement).rjust(34),
        new_file.replace(".shrynk", ""),
    )

print(
    "=" * 80
    + "\n"
    + "total shrynkage: {}% (from {} to {})".format(
        Fore.GREEN + "{}".format(100 - int(new_total / old_total * 100)) + Fore.RESET,
        human_readable_size(old_total, 0),
        human_readable_size(new_total, 0),
    )
)
