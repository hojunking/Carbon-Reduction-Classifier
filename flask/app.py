# main.py

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import carbon_classifier  # Import your inference function

app = Flask(__name__)

# model_inference.py

def infer_from_model(image_path):
    # Your model loading and inference code goes here
    ...
    return prediction


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                'uploads',
                secure_filename(image_file.filename)
            )
            image_file.save(image_location)
            prediction = infer_from_model(image_location)
            return render_template('index.html', prediction=prediction, image_loc=image_file.filename)
    return render_template('index.html', prediction=None, image_loc=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
