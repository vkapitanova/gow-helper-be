from flask import Flask
from flask import request
from opencv import process_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def healthcheck():
    return {"status": "ok"}

@app.route('/upload', methods=['POST', 'PUT'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('out/uploaded_file.jpeg')
        res = process_image('out/uploaded_file.jpeg')
        a = []
        for line in res:
            for val in line:
                a.append(val)
        return {'map': a}
    if request.method == 'PUT':
        content = request.get_data()
        newFile = open("out/uploaded_file.jpeg", "wb")
        # write to file
        newFile.write(content)
        res = process_image('out/uploaded_file.jpeg')
        a = []
        for line in res:
            for val in line:
                a.append(val)
        return {'map': a}

if __name__ == "__main__":
    app.run(debug=True)
