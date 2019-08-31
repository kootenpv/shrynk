import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from shrynk.pandas import infer, PandasCompressor
from flask import Markup
import pandas as pd

from preconvert.output import json  # also install preconvert_numpy

pdc = PandasCompressor("default")
target = "size"
pdc.train_model(target)

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
# max 1 MB
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
             color: red;
         }
         .result.st {
             color: forestgreen;
         }
         .result.nd, .result.rd {
             color: orange;
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
            margin-bottom: 1rem;
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
    <title>Upload CSV File</title>
    <script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>

    <div style='line-height: 1; font-family: "Roboto Slab", serif;'>
        <div class="row">
        <center><h1>shrynk</h1><h4 class="tagline">Use the Power of Machine Learning to Compress</h4></center>
        <div class="container"><div class="row">
          <div class="col s12 m6 l4">
            <div class="card">
                <div class="card-content">
                <span class="card-title">Input: titanic.csv</span>
                <p><code class="codes" style="font-size: smaller;">  y,Pclass,  sex ,age
  0,   3  ,female, 22
  1,   1  ,female, 38
  1,   3  , male , 26
...,  ... ,  ... ,...
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
   "num_obs": 3,
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

</code>
                </p>
              </div>
            </div>
          </div>
        </div>
        <center>
           <h5 style="padding: 1rem;">
             <i class="material-icons icon-flipped">format_quote</i>
             <b>Data & Model</b> for the community, by the community"
             <i class="material-icons">format_quote</i>
           </h5>
        </center>
        </div>
        <center><h5 class="tagline">Usage:</h5></center>
        <div class="container"><div class="row">
          <div class="col s12 m6 l4 offset-l4 offset-m3">
            <center><code class="codes">$ pip install shrynk</center>
</code>
            <code class="codes" style="line-height: 1.3;">>>> from shrynk.pandas import save
>>> save(data_frame, "mydata", optimize="size")
"mydata.csv.bz2"</code>
          </div>
        </div></div>
        <center><h5 class="tagline"> Contribute your data: </h5></center>
        <div class="container">
        <form method=post enctype=multipart/form-data id="form" class="form">
            <div class="row">
              <div class="col s12 m6 l4"><div class="headthree">1. Click below to upload a CSV file and let the compression be predicted.</div></div>
              <div class="col s12 m6 l4"><div class="headthree">2. It will also run all compression methods to see if it is correct.</div></div>
              <div class="col s12 m6 l4  offset-m3"><div class="headthree">3. In case the result is not in line with the ground truth, the features (not the data) will be added to the training data!</div></div>
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
        <div class="container">
    '''


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
                data = pd.read_csv(file)
            except pd.errors.ParserError as e:
                return html + "<h4>{}</h4>".format(str(e))
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            features = pdc.get_features(data)
            inferred = pdc.infer(features)
            features = {
                k.replace("quantile_proportion", "quantile"): round(v, 3)
                if isinstance(v, float)
                else v
                for k, v in features.items()
            }
            bench_res = pdc.run_benchmarks([data], save=False).sort_values(target)
            predicted_compression = " ".join(["{}={!r}".format(k, v) for k, v in inferred.items()])
            result_index = [
                i + 1 for i, x in enumerate(bench_res.index) if x == predicted_compression
            ] + [len(bench_res.index) + 1]
            learning = "none" if result_index[0] == 1 else "inherit"
            nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(result_index[0], str(result_index[0]) + "th")
            return html + str(
                Markup(
                    '<div class="row"><div class="col s12 m6 l3 offset-l2">'
                    + "<b>Filename: </b>"
                    + file.filename
                    + "<br><b>Features: </b>"
                    + '<code class="codes">'
                    + json.dumps(features, indent=4)
                    + "</code>"
                    + '</div>'
                    + '<div class="col s12 m6 l3 offset-l3">'
                    + "<br><center style='line-height: 3'><b>Predicted: </b><br>"
                    # just using features here instead of data to be faster
                    + predicted_compression
                    + "<br><b>Result:</b><br><span class='result {}'>{}</span> / {}<br><div style='display: {}'><span style='color: #ee6e73'>Wrong!</span> We will learn from this...</div>".format(
                        nth[1:], nth, bench_res.shape[0], learning
                    )
                    + "</center></div></div>"
                    + "<center><h4>Ground truth (results of compressing)</h4><div class='show-on-med-and-down hide-on-large-only' style='padding: 0.5rem; color: grey'> -- scroll -> </center>"
                    + bench_res.to_html().replace(
                        '<table border="1" class="dataframe">',
                        '<table id="tableau" border="1" class="dataframe responsive-table highlight z-depth-2" style="line-height: 0.2">',
                    )
                )
            )
        else:
            return html + "<h4>Wrong file type, must have .csv extension.</h4>"
    return html


print("running...")
if __name__ == "__main__":
    app.run("0.0.0.0")
