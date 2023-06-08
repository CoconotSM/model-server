import model.convert_model 
from flask import Flask, jsonify, request
from flask_cors import CORS
import ssl

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

    # TODO: 딥러닝 모델을 사용하여 변환 작업 수행
    output = model.convert_model.final_convert(script_data)

    # 변환 결과를 반환
    output.save('slide1.png')
   
    print(script_data)
    return jsonify({'input': input_text, 'script': script_data})

if __name__ == '__main__':
    app.run(port= 8000, debug=True)



# from flask import Flask, request, jsonify

# app = Flask(__name__)


# @app.route('/api/converto', methods =['POST'])
# def convert():
#     # 리액트에서 받아온 목차, 스크립트 
#     data = request.json
#     input_data = data.get('input')
#     script_data = data.get('script')
    
#     # 변환 모델 코드 작성
#     print('성공')

#     return jsonify({'input': input_data, 'script': script_data})

# if __name__ == '__main__':
#     app.run()

# # from flask import Flask
# # app = Flask(__name__)

# # @app.route('/api/convert')
# # def hello_world():
# #     return 'Hello World!'

# # if __name__ == '__main__':
# #     app.run(port=8000, debug= True) 
