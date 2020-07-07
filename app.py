from flask import Flask, render_template, request, jsonify, send_file
import neural_transfer
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    tensor = neural_transfer.generate()
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = tensor[0]
    img = Image.fromarray(tensor)
    # if np.ndim(tensor)>3:
    #     assert tensor.shape[0] == 1
    #     tensor = tensor[0]
    #     tensor = Image.fromarray(tensor)
    # return tensor
    file_object = io.BytesIO()
    img.save(file_object, 'PNG')
    file_object.seek(0)
    return send_file(file_object, mimetype='image/PNG')

if __name__ == "__main__":
    app.run()