from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/api/convert', methods =['POST'])
def convert():
    # 리액트에서 받아온 목차, 스크립트 
    data = request.json
    input_data = data.get('input')
    script_data = data.get('script')
    
    # 변환 모델 코드 작성

    return jsonify({'input': input_data, 'script': script_data})


if __name__ == '__main__':
    app.run()