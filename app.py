import model.convert_model 
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import ssl
import cv2
from io import BytesIO
import base64

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hello, Flask!"

@app.route('/api/convert', methods=['GET','POST'])
def convert():
    data = request.json
    input_text = data.get('input')
    script_data = data.get('script')
    model_data = data.get('model')
    img_type = data.get('imgtype')


    # TODO: 딥러닝 모델을 사용하여 변환 작업 수행
    output = model.convert_model.final_convert(script_data, input_text, model_data, img_type)
    

    im_file = BytesIO()
    output.save(im_file, format ="PNG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    

   
    return jsonify({'image_url': im_b64.decode('ascii')})

if __name__ == '__main__':
    app.run(port= 8000, debug=True)
    
