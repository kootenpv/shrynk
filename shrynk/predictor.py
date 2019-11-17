import os
import time
import json
import random
import warnings
import pkgutil
import pandas as pd
import numpy as np
import dill
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from shrynk.utils import scalers, get_model_data, shrynk_path, add_z_to_bench
from fractions import Fraction


class Predictor:
    model_name = ""
    model_type = ""

    def __init__(self, model_name="default", clf=RandomForestClassifier, **clf_kwargs):
        if "n_estimators" in clf._get_param_names() and "n_estimators" not in clf_kwargs:
            clf_kwargs["n_estimators"] = 100
        self.clf = clf(**clf_kwargs)
        self.model_data = None
        self.model_name = model_name
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

    def prepare_data(self, size, write, read, scaler, calc_losses):
        targets = ["size", "write_time", "read_time"]
        size_write_read = np.array((size, write, read))
        if self.model_data is None:
            self.model_data = get_model_data(
                self.model_type, self.model_name, self.compression_options
            )
        scale = scalers.get(scaler, scaler)
        features = []
        y = []
        feature_ids = []
        opts = [json.dumps(x) for x in self.compression_options]
        losses = []
        for x in self.model_data:
            vals = [[y[t] for t in targets] for y in x["bench"]]
            fid = x["feature_id"]
            z = (scale(vals) * size_write_read).sum(axis=1).argmin()
            features.append(x["features"])
            y.append(x["bench"][z]["kwargs"])
            feature_ids.append(fid)
            if calc_losses:
                scores = {x: np.nan for x in opts}
                for t in targets:
                    min_t = min([x[t] for x in x["bench"]])
                    for b in x["bench"]:
                        scores[(t, b["kwargs"])] = (b[t] or 1) / (min_t or 1)
                losses.append(scores)
        bests = pd.DataFrame(features)
        bests["y"] = y
        bests["feature_id"] = feature_ids

        if calc_losses:
            return bests, pd.DataFrame(losses)

        return bests

    def validate(self, size, write, read, scaler="z", balanced=True, train_test_ratio=0.66, k=5):
        bests, losses = self.prepare_data(size, write, read, scaler, True)
        bests = pd.concat((bests, losses), axis=1)
        bests = bests.sort_values("y")
        frac = Fraction(train_test_ratio)
        numerator, denominator = frac._numerator, frac._denominator
        results = []
        accs = []
        print()
        print("[shrynk] s={} w={} r={}".format(size, write, read))
        print("----------------")
        for i in range(k):
            bests["train"] = [int(x[:~i], 16) % denominator < numerator for x in bests.feature_id]
            train = bests.query("train")
            test = bests.query("~train")
            if balanced:
                train = self.upsample(train)
                test = self.upsample(test)
            test = test.copy()
            self.clf.fit(train.drop(["y", "train", "feature_id"], axis=1).fillna(-100), train["y"])
            preds = self.clf.predict(test.drop(["y", "train", "feature_id"], axis=1).fillna(-100))
            acc = np.mean(preds == test.y)
            single_scores = test.y.value_counts() / test.y.value_counts().sum()
            single = round(single_scores.max() * 100, 2)
            acc_str = " | accuracy shrynk prediction {}%".format(round(acc * 100, 2))
            it_str = "it {}/{}: ".format(i, k)
            if not balanced:
                strat_str = "accuracy single best strategy {}% ({})".format(
                    single, single_scores.idxmax()
                )
                print(it_str + +acc_str)
            else:
                strat_str = "classes equally weighted, uniform chance: {}%".format(single)
            print(it_str + strat_str + acc_str)
            accs.append(acc)
            test.loc[:, "prediction"] = preds
            dfs = []
            for t in ["size", "read_time", "write_time"]:
                fucks = [test.iloc[i][(t, x)] for i, x in enumerate(test["prediction"])]
                score = pd.Series(fucks).mean()
                scores = [score]
                names = ["shrynk_prediction"]
                for x in losses.columns:
                    if t not in x:
                        continue
                    scores.append(test[x].mean())
                    names.append(x[1])
                dfs.append(pd.DataFrame({t: scores}, index=names))
            result = pd.concat(dfs, axis=1)
            results.append(result)
        # grouping over multiple of the same compressions in the index
        results = pd.concat(results)
        results = results.groupby(results.index).mean()
        if [size, write, read].count(0) == 2:
            if max(size, write, read) == size:
                sorter = "size"
            elif max(size, write, read) == write:
                sorter = "write_time"
            elif max(size, write, read) == read:
                sorter = "read_time"
        else:
            results = add_z_to_bench(results, size, write, read)
            sorter = "z"
        results = results.sort_values(sorter)
        avg_acc = round(np.mean(accs), 3)
        print("Avg Accuracy:", avg_acc)

        print()
        print(
            "results sorted on {}, shown in proportion increase vs ground truth best".format(
                sorter.upper()
            )
        )
        print(results)
        return avg_acc, results

    def train_model(self, size, write, read, scaler="z", balanced=True):
        size_write_read = np.array((size, write, read))
        model_path = "{}_{}_{}_{}_{}.pkl".format(self.model_type, self.model_name, *size_write_read)
        try:
            with open(shrynk_path(model_path), "rb") as f:
                self.clf = dill.load(f)
            print("loaded model pickle")
            return self.clf
        except FileNotFoundError:
            bests = self.prepare_data(size, write, read, scaler, False)
            if balanced:
                bests = self.upsample(bests)
            self.clf.fit(bests.drop(["y", "feature_id"], axis=1).fillna(-100), bests.y)
        self.clf.columns_ = pd.DataFrame([x["features"] for x in self.model_data]).columns
        with open(shrynk_path(model_path), "wb") as f:
            dill.dump(self.clf, f)
            print("saved model pickle")
        return self.clf

    def _predict(self, features, deserialize=True):
        if isinstance(features, dict):
            features = pd.DataFrame([features], columns=self.clf.columns_)
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
        pred = self.clf.predict(features.fillna(-100))[0]
        if not isinstance(pred, str):
            pred = pred[0]
        if deserialize:
            from preconvert.output import json

            return json.loads(pred)
        return pred

    def predict(self, data, deserialize=True):
        features = self.get_features(data)
        return self._predict(features, deserialize)

    def predict_proba(self, features):
        if isinstance(features, pd.DataFrame):
            features = self.get_features(features)
        if isinstance(features, dict):
            features = pd.DataFrame([features], columns=self.columns_)
        return dict(zip(self.clf.classes_, self.clf.predict_proba(features)[0]))

    infer = predict

    # def save_model(self):

    # def load_model(self):
    # def eval_model(self):
