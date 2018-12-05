

from flask import Flask, request, redirect, render_template, url_for
import os
from werkzeug.utils import secure_filename
import edge_detectors

UPLOAD_FOLDER = 'C:/Users/MBAdmin/PycharmProjects/FlaskApp'
ALLOWED_EXTENSIONS = set(['.png', '.jpg','.jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/bacon", methods=['GET', 'POST'])
def bacon():
    if request.method == 'POST':
        return "You are using POST"
    else:
        return "You are probably using GET"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def show_home():
    return render_template('index.html')


@app.route('/upload')
def upload_file_page():
    return render_template('index.html')

@app.route('/canny')
def render_canny():
    return render_template('canny.html')

@app.route('/sobel')
def render_sobel():
    return render_template('sobel.html')

@app.route('/prewitt')
def render_prewitt():
    return render_template('prewitt.html')

@app.route('/histogram_equalization')
def render_histeq():
    return render_template('histogram_equalization.html')

@app.route('/laplacian')
def render_laplacian():
    return render_template('laplacian.html')


@app.route('/uploader_canny', methods=['GET', 'POST'])
def upload_canny():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      edge_detectors.canny_edge_detector(f.filename)
      return render_template('upload_successful.html')

@app.route('/uploader_sobel', methods=['GET', 'POST'])
def upload_sobel():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      edge_detectors.sobel(f.filename)
      return render_template('upload_successful.html')

@app.route('/uploader_prewitt', methods=['GET', 'POST'])
def upload_prewitt():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      edge_detectors.prewitt(f.filename)
      return render_template('upload_successful.html')


@app.route('/uploader_histeq', methods=['GET', 'POST'])
def upload_histeq():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      edge_detectors.histogram_equalization(f.filename)
      return render_template('upload_successful.html')


@app.route('/uploader_laplacian', methods=['GET', 'POST'])
def upload_laplacian():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      edge_detectors.laplac(f.filename)
      return render_template('upload_successful.html')



@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully ' + f.filename



@app.route('/profile/<name>')
def profile(name):
    return render_template("profile.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)