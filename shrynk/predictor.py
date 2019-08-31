import os
import random
import warnings
import json
import pkgutil
import pandas as pd
import numpy as np
import dill
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier


class Predictor:
    def __init__(self, model_name, clf=RandomForestClassifier, **clf_kwargs):
        self.clf = clf(**clf_kwargs)
        self.model_name = model_name
        self.value_lookup = None
        self.arg_lookup = None
        self.X_ = None
        self.y_ = None

    def get_model_data(self):
        """ Gets the model data """
        try:
            data = pkgutil.get_data("data", "shrynk/{}.jsonl".format(self.model_name.lower()))
            data = pd.DataFrame(
                [json.loads(line) for line in data.decode("utf8").split("\n") if line.strip()]
            )
            print("from package")
        except FileNotFoundError:
            with open(os.path.expanduser("~/shrynk_{}.jsonl".format(self.model_name))) as f:
                data = pd.DataFrame([json.loads(line) for line in f if line.strip()])
        data = pd.concat((data, pd.io.json.json_normalize(data["result"])), axis=1)
        return data

    def upsample(self, bests):
        largest_n = bests["kwargs"].astype(str).value_counts().max()
        bests = (
            bests.groupby("y")
            .apply(lambda x: x.sample(largest_n, replace=True))
            .set_index("group_id")
        )
        return bests

    def train_test_split(self, bests, train_ratio=0.5):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for name, group in bests.groupby("group_id"):
            if random.random() < train_ratio:
                X_train.append(pd.io.json.json_normalize(group["features"]).set_index(group.index))
                y_train.append(group.y)
            else:
                X_test.append(pd.io.json.json_normalize(group["features"]).set_index(group.index))
                y_test.append(group.y)
        X_train = pd.concat(X_train)
        y_train = pd.concat(y_train)
        X_test = pd.concat(X_test)
        y_test = pd.concat(y_test)
        return X_train, y_train, X_test, y_test

    def crossval(self, data, bests, target, n=5, train_ratio=0.5):
        results = defaultdict(list)
        for i in range(n):
            print("crossval", i + 1, "/", n)
            X_train, y_train, X_test, y_test = self.train_test_split(bests, train_ratio)
            return X_train, y_train, X_test, y_test
            self.clf.fit(X_train.fillna(-100), y_train)
            preds = self.clf.predict(X_test.fillna(-100))
            class_accuracy = np.mean(preds == y_test)
            # # # # self.X_, self.y_ = X, y
            print("predicting")
            for g_id, p, t in zip(X_test.index, preds, y_test):
                a = self.arg_lookup[p]
                b = self.arg_lookup[t]
                g = data[data.group_id == g_id]
                try:
                    ga = g[g.kwargs == a]
                    gb = g[g.kwargs == b]
                    results["pred"].extend(np.array(ga[target]) / np.array(gb[target]))
                except ValueError:
                    continue
            strategies = y_train.unique()
            for strategy in strategies:
                print("strategy", strategy)
                preds = [strategy] * len(y_test)
                for g_id, p, t in zip(X_test.index, preds, y_test):
                    a = self.arg_lookup[p]
                    b = self.arg_lookup[t]
                    g = data[data.group_id == g_id]
                    ga = g[g.kwargs == a]
                    gb = g[g.kwargs == b]
                    if ga.empty:
                        continue
                    results[strategy].extend(np.array(ga[target]) / np.array(gb[target]))
        report = defaultdict(dict)
        pred_mean = np.mean(results["pred"])
        for strategy in strategies:
            if strategy == "pred":
                continue
            strat_mean = np.mean(results[strategy])
            savings = (strat_mean - pred_mean) / pred_mean * 100
            strategy_name = str(self.arg_lookup[strategy])
            report[strategy_name] = savings
        return class_accuracy, report

    def train_model(self, target, n_validations=None):
        data = self.get_model_data()
        bests = data.groupby("group_id").apply(lambda group: group.loc[group[target].idxmin()])
        # have to encode it into integers and do a lookup here when wrapping prediction!
        # then return actual best args
        self.value_lookup = {
            x: i for i, x in enumerate(set([tuple(x.items()) for x in bests["kwargs"]]))
        }
        self.arg_lookup = {i: dict(x) for x, i in self.value_lookup.items()}
        y = pd.Series(
            [self.value_lookup[tuple(x.items())] for x in bests["kwargs"]], index=bests.index
        )
        bests["y"] = y
        bests = self.upsample(bests)
        if n_validations is not None:
            return self.crossval(data, bests, target, n_validations)
        try:
            clf = dill.loads(
                pkgutil.get_data("data", "shrynk/{}_{}.pkl".format(self.model_name.lower(), target))
            )
            self.clf = clf
        except FileNotFoundError:
            self.clf.fit(pd.io.json.json_normalize(bests["features"]).fillna(-100), bests.y)
        return self.clf

    def predict(self, features):
        if isinstance(features, pd.DataFrame):
            features = self.get_features(features)
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        return self.arg_lookup[self.clf.predict(features.fillna(-100))[0]]

    infer = predict

    # def save_model(self):

    # def load_model(self):
    # def eval_model(self):
