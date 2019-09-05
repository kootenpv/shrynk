import io
import re
import os
import sys
import time
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
import shrynk
from shrynk.pandas import infer, PandasCompressor
from shrynk.utils import md5, scalers

# from shrynk.utils import md5
from flask import Markup
import pandas as pd

from preconvert.output import json  # also install preconvert_numpy

from google.cloud import storage

BUCKET_NAME = "api-project-435023019049.appspot.com"
IN_PRODUCTION = "main.py" != sys.argv[0]
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/pascal/Downloads/api-project-435023019049-a83be40d22b6.json"


def get_blob(features):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob_name = md5(features)
    blob = bucket.blob(blob_name + "_shrynk_" + shrynk.__version__)
    print("accessed blob")
    return blob
    # blob.upload_from_string(json.dumps(results))
    # print("yesssssss", json.loads(blob.download_as_string()))


weights = np.array((3, 1, 1))
scale = scalers["z"]
pdc = PandasCompressor("default", n_estimators=50)
pdc.train_model(*weights, scale)

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'parquet'}
ALLOWED_EXTENSIONS.update([x["compression"] for x in pdc.compression_options])

app = Flask(__name__)
# max 1 MB
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n - 1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string = before + after
    return new_string


html = '''
    <!doctype html>
    <head>
        <style>
         body {
           color: #1976d2;
         }
         code {
           color: black;
         }
         .result {
             font-size: larger;
             color: #ff9999;
         }
         .result.st {
             color: darkseagreen;
         }
         .result.nd, .result.rd {
             color: orange;
         }
         .resultinv {
             font-size: larger;
             background-color: #ff9999;
         }
         .resultinv.st {
             background-color: darkseagreen;
         }
         .resultinv.nd, .resultinv.rd {
             background-color: orange;
         }
         .headthree {
             border: 1px dashed;
             padding: 1rem;
             border-radius: 10px;
             height: 80px;
         }
         .tagline {
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin-top: 1rem;
            margin-bottom: 2rem;
            background-color: #ee6e73 !important;
            color: #fff !important;
         }
         .codes {
             word-break: normal;
             word-wrap: normal;
             white-space: pre;
         }
        .icon-flipped {
            transform: scaleX(-1);
            -moz-transform: scaleX(-1);
            -webkit-transform: scaleX(-1);
            -ms-transform: scaleX(-1);
        }
        </style>

        <!--Import Google Icon Font-->
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link rel="stylesheet" type="text/css" href="https://fonts.googleapis.com/css?family=Roboto:200|Roboto+Slab">
        <!--Import materialize.css-->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">

        <!--Let browser know website is optimized for mobile-->
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    </head>
    <title>shrynk.ai - Using the Power of Machine Learning to Compress</title>
    <script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>

    <div style='line-height: 1; font-family: "Roboto Slab", serif;'>
        <div class="row">
         <a href="https://github.com/kootenpv/shrynk" class="github-corner" aria-label="View source on GitHub"><svg width="80" height="80" viewBox="0 0 250 250" style="fill:#ee6e73; color:#fff; position: absolute; top: 0; border: 0; right: 0;" aria-hidden="true"><path d="M0,0 L115,115 L130,115 L142,142 L250,250 L250,0 Z"></path><path d="M128.3,109.0 C113.8,99.7 119.0,89.6 119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3 123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9 134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;" class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5 119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0 127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4 163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6 187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2 216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4 203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1 C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7 141.6,141.9 141.8,141.8 Z" fill="currentColor" class="octo-body"></path></svg></a><style>.github-corner:hover .octo-arm{animation:octocat-wave 560ms ease-in-out}@keyframes octocat-wave{0%,100%{transform:rotate(0)}20%,60%{transform:rotate(-25deg)}40%,80%{transform:rotate(10deg)}}@media (max-width:500px){.github-corner:hover .octo-arm{animation:none}.github-corner .octo-arm{animation:octocat-wave 560ms ease-in-out}}</style>
         <center>
            <a href="/"><img src="https://i.ibb.co/5L5QC2K/Shrynk.png" width="300rem" style="padding: 1rem" /></a>
        </center>

        <!-- <hr style="border-top: 1px solid red;" class="z-depth-2"> -->

        <center>
            <h4 class="tagline">Use the Power of Machine Learning to Compress</h4>

            <p>Why choose a compression method yourself when machine learning can predict the best one for your data and requirements?</p>

            <h5 style="padding: 1rem;">
                How it works in 3 steps:
            </h5>
        </center>


        <div class="container"><div class="row">
          <div class="col s12 m6 l4">
            <div class="card">
                <div class="card-content">
                <span class="card-title">Input: titanic.csv</span>
                <p><code class="codes" style="font-size: smaller;">  y,Pclass,  sex ,age
  0,   3  ,female, 22
  1,   1  , male , 38
  0,   2  , male , 30
  1,   3  ,female, 26 [x97]
                  </code>
                </p>
              </div>
            </div>
          </div>
          <div class="col s12 m6 l4">
            <div class="card">
                <div class="card-content">
                <span class="card-title">Featurize</span>
                <p><code class="codes" style="font-size: smaller;">{
   "num_obs": 100,
   "num_cols": 4,
   "num_str": 1,
   ...
}</code>
                </p>
              </div>
            </div>
          </div>
          <div class="col s12 m6 l4 offset-m3">
            <div class="card">
                <div class="card-content">
                <span class="card-title">Model Predictions</span>
                <p><code class="codes" style="font-size: smaller;">
Smallest size:      csv+bz2   ✓
Fastest write time: csv+gzip  ✓
Fastest read time:  csv+gzip  ✓

Weighted (3, 1, 1): csv+bz2   ✓
</code>
                </p>
              </div>
            </div>
          </div>
        </div>
        </div>


        <center><h5 class="tagline">Usage:</h5></center>
        <div class="container"><div class="row">
          <div class="col s12 m6 l4 offset-l4 offset-m3">
<center style="margin-bottom: 1rem">Install:</center>
            <center><code class="codes">$ pip install shrynk</center></code>
<p>Now in python...</p>
            <code class="codes" style="line-height: 1.3;">>>> from shrynk.pandas import save</code>
<p>You control how important size, write_time and read_time are. <br> Here, size is 3 times more important than write and read. </p>
<code class="codes show-on-med-and-down hide-on-large-only" style="line-height: 1.3;">>>> save(df, "mydata",
         size=3, write=1, read=1)
"mydata.csv.bz2"</code>
<code class="codes hide-on-med-and-down show-on-large" style="line-height: 1.3;">>>> save(df, "mydata", size=3, write=1, read=1)
"mydata.csv.bz2"</code>
          </div>
        </div></div>

        <center><h5 class="tagline"> Contribute your data: </h5></center>
        <div class="container">
<center>
           <h5 style="padding: 1rem;">
             <i class="material-icons icon-flipped">format_quote</i>
             <b>Data & Model</b> for the community, by the community"
             <i class="material-icons">format_quote</i>
           </h5>
        </center>
        <form action="/" method=post enctype=multipart/form-data id="form" class="form">
            <div class="row">
              <div class="col s12 m6 l4" style="margin-top: 1rem; margin-bottom: 1rem"><div class="headthree">1. Click below to upload a CSV file and let the compression be predicted.</div></div>
              <div class="col s12 m6 l4" style="margin-top: 1rem; margin-bottom: 1rem"><div class="headthree">2. It will also run all compression methods to see if it is correct.</div></div>
              <div class="col s12 m6 l4 offset-m3" style="margin-top: 1rem; margin-bottom: 1rem"><div class="headthree">3. In case the result is not in line with the ground truth, the features (not the data) will be added to the training data!</div></div>
            </div>
            <center>
          <center>
            <div class="upload-btn-wrapper" style="position: relative; overflow: hidden; display: inline-block;">
              <button class="btn z-depth-2" style="border: 2px solid #ee6e73; color: #ee6e73; background-color: white; padding: 8px 20px; border-radius: 8px; font-size: 20px; font-weight: bold; height: 60px; margin-top: 1rem; margin-bottom: 1rem;">
                Upload file... <span style="font-size: small;">(max 1 MB.)</span>
              </button>
              <input type="file" id="file" name="file" style="font-size: 100px; position: absolute; left: 0; top: 0; opacity: 0;"/>
            </div>
          </center>
          <input type="submit" value="Upload" style="visibility: hidden;"/>
          <h5 id="loading" style="display: none">Loading...</h5>
        </form>
        <div style="margin-bottom: 2rem; margin-top: -1.5rem;">or try the <a href="/example" style="text-decoration:underline">example</a></div>
        </div>
        <div id="modal1" class="modal">
          <div class="modal-content">
            <h4>Loading...</h4>
            <p>should be over soon ;)</p>
          </div>
        </div>
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', function() {
              var elems = document.querySelectorAll('.modal');
              var instances = M.Modal.init(elems);
              window.scrollTo(0,document.getElementById("tableau").scrollHeight);
            });
            document.getElementById("file").onchange = function() {
                var instance = M.Modal.getInstance(document.getElementById("modal1"));
                instance.open();
                document.getElementById("form").submit();
            };
        </script>
    '''


def get_benchmark_html(df, fname):
    features = pdc.get_features(df)
    bench_res = None
    if IN_PRODUCTION:
        blob = get_blob(features)
        if blob.exists():
            results = json.loads(blob.download_as_string())
            bench_res = results["bench"]
        else:
            results = pdc.run_benchmarks([df], save=False, ignore_seen=False, timeout=False)[0]
            bench_res = results["bench"]
            blob.upload_from_string(json.dumps(results))
    else:
        bench_res = pdc.run_benchmarks([df], save=False, ignore_seen=False, timeout=False)[0][
            "bench"
        ]
    kwargs = [x.pop("kwargs") for x in bench_res]
    bench_res = pd.DataFrame(bench_res, index=kwargs)
    inferred = pdc.infer(features)
    z_name = "z {}".format(tuple(weights))
    bench_res[z_name] = (scale(bench_res) * weights).sum(axis=1)
    bench_res = bench_res.round(5).sort_values(z_name)
    bench_res = bench_res[[z_name, "size", "write_time", "read_time"]]
    y = json.dumps(inferred)
    res_index = [i + 1 for i, x in enumerate(bench_res.index) if x == y] + [-1]
    bench_res.index = [
        " ".join(["{}={!r}".format(k, v) for k, v in json.loads(x).items()])
        for x in bench_res.index
    ]
    learning = "none" if res_index and res_index[0] == 1 else "inherit"
    nth = {1: "1st", 2: "2nd", 3: "3rd", -1: "999"}.get(res_index[0], str(res_index[0]) + "th")
    # upload(features, "{}-{}".format(file.filename, time.time()))
    features = {
        k.replace("quantile_proportion", "quantile"): round(v, 3) if isinstance(v, float) else v
        for k, v in features.items()
    }
    return str(
        Markup(
            '<center> <h5 class="tagline"> Results: </h5></center>'
            + '<div class="container" style="margin-top: 2rem"><div class="row">'
            + '<div class="col s12 m6 l3 offset-l2">'
            + "<b>Filename: </b>"
            + fname
            + "<br><b>Features: </b>"
            + '<code class="codes">'
            + json.dumps(features, indent=4)
            + "</code>"
            + '</div>'
            + '<div class="col s12 m6 l3 offset-l3">'
            + "<br><center style='line-height: 3'><b>Predicted: </b><br>"
            # just using features here instead of data to be faster
            + " ".join(["{}={!r}".format(k, v) for k, v in inferred.items()])
            + "<br><b>Result:</b><br><span class='result {}'>{}</span> / {}<br><div style='display: {}'><span style='color: #ee6e73'>Wrong!</span> We will learn from this...</div>".format(
                nth[1:], nth, bench_res.shape[0], learning
            )
            + "</center></div></div>"
            + "<center><h4>Ground truth (results of compressing)</h4><div class='show-on-med-and-down hide-on-large-only' style='padding: 0.5rem; color: grey'> -- scroll -> </center>"
            + replacenth(
                bench_res.to_html().replace(
                    '<table border="1" class="dataframe">',
                    '<table id="tableau" border="1" class="dataframe responsive-table highlight z-depth-2" style="line-height: 0.2">',
                ),
                "<tr>",
                '<tr class="resultinv {}">'.format(nth[1:]),
                int(nth[:-2]),
            )
        )
    )


res = []


def lazy_titanic():
    if res:
        return res[0]
    example_data = pd.read_csv(
        io.StringIO(
            """y,Pclass,  sex ,age
  0,   3  ,female, 22
  1,   1  , male , 38
  0,   2  , male , 30
"""
            + "1,   3  ,female, 26\n" * 97
        )
    )
    res.append(html + get_benchmark_html(example_data, "titanic_example.csv"))
    return res[0]


@app.route('/example', methods=['GET', 'POST'])
def run_example():
    return lazy_titanic()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return html + "<h4>No file part</h4>"
        file = request.files['file']
        if file.filename == '':
            return html + "<h4>No selected file</h4>"
        if file and allowed_file(file.filename):
            try:
                infer = pdc.infer_from_path(file.filename)
                data = pdc.load(file, infer)
            except pd.errors.ParserError as e:
                return html + "<h4>{}</h4>".format(str(e))
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return html + get_benchmark_html(data, file.filename)
        else:
            return html + "<h4>Wrong file type, cannot read. Try a .csv file. </h4>"
    return html


if __name__ == "__main__":
    app.run("0.0.0.0")
