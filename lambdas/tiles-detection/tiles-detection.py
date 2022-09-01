import json
import cv2 as cv
import math
import numpy as np
import base64


def handler(event, context):
    content = base64.b64decode(event["body"])
    query_params = (event["queryStringParameters"])
    x = query_params["x"]
    y = query_params["y"]
    grid_size = query_params["grid_size"]
    res = process_image(content, int(x), int(y), int(grid_size))
    print(res)
    a = []
    for line in res:
        for val in line:
            a.append(val)
    return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps({'map': a}),
    }


def process_image(img_string, grid_x, grid_y, grid_size):
    nparr = np.fromstring(img_string, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    print(img.shape)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grid_img = img_gray[grid_y:grid_y + grid_size, grid_x:grid_x + grid_size]
    # cv.imwrite('out/grid_extracted.jpeg', grid_img)
    oh, ow = grid_img.shape
    perfect_rectangle_size = 1200
    scale = perfect_rectangle_size / oh
    nw, nh = ow * scale, oh * scale
    resized = cv.resize(grid_img, (math.ceil(nw), math.ceil(nh)), interpolation=cv.INTER_AREA)
    # cv.imwrite('out/resized.jpeg', resized)
    elem_size = 150
    map = np.zeros((8, 8), dtype=np.dtype('U24'))
    for i in range(8):
        for j in range(8):
            map[i][j] = 'UN'

    for i in range(8):
        for j in range(8):
            # print(i, j)
            elem = resized[i*elem_size:(i+1)*elem_size-1, j*elem_size:(j+1)*elem_size-1]
            # cv.imwrite('temp/elem{}{}.jpeg'.format(i, j), elem)
            max_template = ''
            max_match = 0
            templates_map = {'yellow.jpeg': 'basic_yellow', 'green.jpeg': 'basic_green', 'red.jpeg': 'basic_red', 'blue.jpeg': 'basic_blue',
                             'brown.jpeg': 'basic_brown', 'violet.jpeg': 'basic_violet', 'skull.jpeg': 'skull_normal', 'rock_skull.jpeg': 'skull_rock',
                             'block.jpeg': 'block', 'great_yellow.jpeg': 'great_yellow', 'great_green.jpeg': 'great_green',
                             'great_red.jpeg': 'great_red', 'great_blue.jpeg': 'great_blue',
                             'great_brown.jpeg': 'great_brown', 'great_violet.jpeg': 'great_violet'}
            for template_name in ['yellow.jpeg', 'green.jpeg', 'red.jpeg', 'blue.jpeg', 'brown.jpeg', 'violet.jpeg', 'skull.jpeg', 'rock_skull.jpeg', 'block.jpeg',
                                  'great_yellow.jpeg', 'great_green.jpeg', 'great_red.jpeg', 'great_blue.jpeg', 'great_brown.jpeg', 'great_violet.jpeg']:
                # print('matching {}'.format(template_name))
                template = cv.imread('/var/task/templates/{}'.format(template_name), cv.IMREAD_COLOR)
                tmp_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
                res = cv.matchTemplate(elem, tmp_gray, cv.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                # print('result', max_val)
                if max_val > max_match:
                    max_template = template_name
                    max_match = max_val
            if max_match < 0.5:
                print("low match", max_template, max_match)
                map[i][j] = 'UN'
            else:
                map[i][j] = templates_map[max_template]
    for i in range(8):
        res = ''
        for j in range(8):
            res += '{} '.format(map[i][j])
        print(res)
    return map
