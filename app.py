import model.convert_model 
from flask import Flask, jsonify, request
from flask_cors import CORS
import ssl
import base64
from io import BytesIO
from PIL import Image

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

    # 딥러닝 모델을 사용하여 변환 작업 수행
    output = model.convert_model.final_convert(input_text, script_data)

    # PIL image to base64
    im_file = BytesIO()
    output.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    # output.save('slide1.png')
   
    # base64 to ascii
    return jsonify({'image_url': im_b64.decode('ascii')})

if __name__ == '__main__':
    app.run(port= 8000, debug=True)