from flask import Flask, redirect, jsonify, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import request
import os
import my_yolov6
import cv2

app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Check uploaded files
ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize your model: best_ckpt.pt (can be replaced by your model file)
yolov6_model = my_yolov6.my_yolov6("weights/best_ckpt.pt", "cpu", "data/mydataset.yaml", 640, False)

@app.route('/', methods=['POST', 'GET'])
def predict_yolov6():
    if request.method == 'POST':
        if 'img' in request.files:
            b_box = {}
            images = request.files.getlist('img')
            # Save files
            for image in images:
                imagename = secure_filename(image.filename)
                if allowed_file(imagename):
                    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], imagename)
                    image.save(path_to_save)
                    frame = cv2.imread(path_to_save)

                    # Detect via YOLOv6 model
                    frame, no_object, bounding_box = yolov6_model.infer(frame, path_img=path_to_save)
                    if no_object > 0:
                        cv2.imwrite(path_to_save, frame)
                    del frame
                    
                    # Freeing files after processing
                    try:
                        os.remove(path_to_save)
                    except OSError as e:
                        return "Error: %s - %s." % (e.filename, e.strerror)

                    # Map an image to all its bounding box information
                    b_box[imagename.split(".")[0]] = bounding_box
                else:
                    b_box[imagename.split(".")[0]] = [{ "message" : "the file format is invalid" }]

            # This is a JSON - { "image_name": [ { "name" : "disease_name and probability", "coordinates" : [coordinates of two points] }, ... ] }
            return jsonify(b_box)
        else:
            return jsonify({ "message" : "not found" })
    else:
        return jsonify({ "message" : "wrong methods" })

@app.route('/url/<imgname>', methods=['GET'])
def url_json(imgname):
    if request.method == 'GET':
        return jsonify({ "path" : "{}".format(imgname) })
    else:    
        return jsonify({ "message" : "wrong methods" })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)