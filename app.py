from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
import neural_transfer
from PIL import Image
import numpy as np
import io
import os

app = Flask(__name__)
app.debug = True

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'img')


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method=='post'   :
        contentfile= request.files['contentfile']
        stylefile = request.files['stylefile']
    return render_template("index.html")

@app.route('/result')
def result():
    tensor = neural_transfer.generate()
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = tensor[0]
    img = Image.fromarray(tensor)
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')
    # return "LETS GO"

@app.route('/view-image', methods=["GET", "POST"])
def uploaded_image():
    if request.method == 'POST':
        if request.files:
            print(" I have files")
            contentfile = request.files["contentfile"]
            stylefile = request.files["stylefile"]
            content_filename = os.path.join(UPLOAD_FOLDER, 'contentfile.png')
            contentfile.save(content_filename)
            style_filename = os.path.join(UPLOAD_FOLDER, 'stylefile.png')
            stylefile.save(style_filename)
            print("Image saved")
            return redirect("/result")
            # return render_template("index.html")
    return render_template("index.html")