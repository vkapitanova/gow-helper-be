from flask import Flask, request
from flask import request
from opencv import process_image, detect_grid
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

@app.route('/')
def healthcheck():
    return {"status": "ok"}

@app.route('/upload', methods=['PUT'])
def upload_file():
    args = request.args
    x = args.get('x')
    y = args.get('y')
    grid_size = args.get('grid_size')
    print(x, y, grid_size)
    content = request.get_data()
    file = open("out/uploaded_file.jpeg", "wb")
    file.write(content)
    res = process_image('out/uploaded_file.jpeg', int(x), int(y), int(grid_size))
    a = []
    for line in res:
        for val in line:
            a.append(val)
    return {'map': a}


@app.route('/detect-grid', methods=['PUT'])
def detect_grid_handler():
    content = request.get_data()
    res_file, x, y, grid_size, my_mana, opponent_mana = detect_grid(content)
    image_file = open(res_file, "rb")
    encoded_grid = base64.b64encode(image_file.read())

    # return {'grid': encoded_grid.decode(), 'x': x, 'y': y, 'grid_size': grid_size}
    return {'x': x, 'y': y, 'grid_size': grid_size, 'my_mana': my_mana, 'opponent_mana': opponent_mana}


if __name__ == "__main__":
    app.run(debug=True)
