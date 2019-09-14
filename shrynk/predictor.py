import time
import os
import random
import warnings
import pkgutil
import pandas as pd
import numpy as np
import dill
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from shrynk.utils import scalers, get_model_data


class Predictor:
    def __init__(self, model_name, clf=RandomForestClassifier, **clf_kwargs):
        if "n_estimators" in clf._get_param_names() and "n_estimators" not in clf_kwargs:
            clf_kwargs["n_estimators"] = 100
        self.clf = clf(**clf_kwargs)
        self.model_name = model_name
        self.model_data = None
        self.X_ = None
        self.y_ = None

    def upsample(self, bests):
        largest_n = bests["y"].astype(str).value_counts().max()
        bests = (
            bests.groupby("y")
            .apply(lambda x: x.sample(largest_n, replace=True))
            .reset_index(drop=True)
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

    def train_model(self, size, write, read, scaler="z", validate_clfs=False, balanced=True):
        targets = ["size", "write_time", "read_time"]
        if self.model_data is None:
            self.model_data = get_model_data(self.model_name, self.compression_options)
        scale = scalers.get(scaler, scaler)
        size_write_read = np.array((size, write, read))
        features = []
        y = []
        feature_ids = []
        for x in self.model_data:
            vals = [[y[t] for t in targets] for y in x["bench"]]
            z = (scale(vals) * size_write_read).sum(axis=1).argmin()
            features.append(x["features"])
            y.append(x["bench"][z]["kwargs"])
            feature_ids.append(x["feature_id"])

        bests = pd.DataFrame(features)
        bests["y"] = y
        bests["feature_id"] = feature_ids

        bests = bests.sort_values("y")
        bests["train"] = bests.index % 2 == 0

        if balanced:
            bests = self.upsample(bests)
        if validate_clfs:
            train = bests.query("train")
            test = bests.query("~train")
            self.clf.fit(train.drop(["y", "train", "feature_id"], axis=1).fillna(-100), train["y"])
            preds = self.clf.predict(test.drop(["y", "train", "feature_id"], axis=1).fillna(-100))
            print("accuracy", np.mean(preds == test.y), "now refitting...")
        self.clf.fit(bests.drop(["y", "train", "feature_id"], axis=1).fillna(-100), bests.y)
        # TO FIX CROSSVAL
        # if n_validations is not None:
        #     return self.crossval(data, bests, target, n_validations)
        try:
            weight = "{}_{}_{}".format(*size_write_read)
            clf = dill.loads(
                pkgutil.get_data("data", "shrynk/{}_.pkl".format(self.model_name.lower(), weight))
            )
            self.clf = clf
            print("loaded model pickle")
        except FileNotFoundError:
            self.clf.fit(bests.drop(["y", "train", "feature_id"], axis=1).fillna(-100), bests.y)
        return self.clf

    def predict(self, features):
        from preconvert.output import json

        if isinstance(features, pd.DataFrame):
            features = self.get_features(features)
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        pred = self.clf.predict(features.fillna(-100))[0]
        if not isinstance(pred, str):
            pred = pred[0]
        return json.loads(pred)

    def predict_proba(self, features):
        if isinstance(features, pd.DataFrame):
            features = self.get_features(features)
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        return dict(zip(self.clf.classes_, self.clf.predict_proba(features)[0]))

    infer = predict

    # def save_model(self):

    # def load_model(self):
    # def eval_model(self):
