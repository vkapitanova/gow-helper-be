import json
import cv2 as cv
import math
import numpy as np
import base64


def handler(event, context):
    content = base64.b64decode(event["body"])
    grid_bytes, x, y, grid_size = detect_grid(content)
    gid_base64 = base64.b64encode(grid_bytes).decode()

    print(json.dumps({'grid': gid_base64, 'x': x, 'y': y, 'grid_size': grid_size}))
    return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps({'grid': gid_base64, 'x': x, 'y': y, 'grid_size': grid_size}),
    }


def detect_grid(img_string):
    nparr = np.fromstring(img_string, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    oh, ow, d = img.shape
    perfect_rectangle_size = 1200
    scale = perfect_rectangle_size / oh
    nw, nh = ow * scale, oh * scale
    resized = cv.resize(img, (math.ceil(nw), math.ceil(nh)), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(img_gray,cv.HOUGH_GRADIENT,1,100,param1=50,param2=70,minRadius=50,maxRadius=75)
    circles = np.uint16(np.around(circles))
    coords = circles[0,:]
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0
    for c in coords:
        x, y, r = c
        if x - r < min_x:
            min_x = x - r
        if x + r > max_x:
            max_x = x + r
        if y - r < min_y:
            min_y = y - r
        if y + r > max_y:
            max_y = y + r
    grid_size = max(max_x - min_x, max_y - min_y)

    res = resized[min_y:min_y + grid_size, min_x:min_x + grid_size]
    grid_bytes = cv.imencode('.jpg', res)[1]
    return grid_bytes, round(min_x / scale), round(min_y / scale), round(grid_size / scale)
