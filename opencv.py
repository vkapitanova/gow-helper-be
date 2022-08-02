import cv2 as cv
import math
import numpy as np


def find_template_matches(img, template_name, threshold):
    original_template_size = 141
    template = cv.imread('templates/{}'.format(template_name), 0)

    resized_temp = template
    best_matches = []
    max_value = 0
    max_value_location = []
    best_run = -1
    for i in range(20):
        # print(i)
        all_matches = []
        tw, th = resized_temp.shape[::-1]
        img_attempt = img.copy()
        res = cv.matchTemplate(img_attempt, resized_temp, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        loc = np.where(res >= threshold)
        x, y = loc
        # print(len(x))
        if len(x) > 0:
            # print("matches found")
            for pt in zip(*loc[::-1]):
                all_matches.append([pt[0], pt[1], tw, template_name])
                cv.rectangle(img_attempt, pt, (pt[0] + tw, pt[1] + th), (255, 255, 255), 2)
            if max_val > max_value:
                max_value = max_val
                max_value_location = max_loc
                best_run = i
                best_matches = all_matches
            if max_value != 0 and max_val < max_value:
                break
            # cv.imwrite('out/res{}.png'.format(i), img_attempt)
        resized_temp = cv.resize(template, (original_template_size-i*2, original_template_size-i*2), interpolation=cv.INTER_AREA)
        # cv.imwrite('out/scaled{}.png'.format(i), resized_temp)
    # print(max_value, max_value_location, best_run)
    # cleaning close points
    cleaned = best_matches
    i = 0
    # print(cleaned)
    while i < len(cleaned):
        m1 = cleaned[i]
        temp = [m1]
        for j, m2 in enumerate(cleaned):
            if m1 == m2:
                continue
            distance = math.sqrt((m1[0]-m2[0])*(m1[0]-m2[0]) + (m1[1]-m2[1])*(m1[1]-m2[1]))
            if distance > 30:
                temp.append(m2)
        cleaned = temp
        i = i + 1

    return cleaned

def add_matches(img, matches):
    # print(matches)
    for pt in matches:
        cv.rectangle(img, [pt[0], pt[1]], (pt[0] + pt[2], pt[1] + pt[2]), (255, 255, 255), 2)
    cv.imwrite('out/res.png', img)


def detect_grid(img_file):
    img = cv.imread(img_file, cv.IMREAD_COLOR)
    oh, ow, d = img.shape
    perfect_rectangle_size = 1200
    scale = perfect_rectangle_size / oh
    nw, nh = ow * scale, oh * scale
    resized = cv.resize(img, (math.ceil(nw), math.ceil(nh)), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(img_gray,cv.HOUGH_GRADIENT,1,100,param1=50,param2=70,minRadius=50,maxRadius=75)
    circles = np.uint16(np.around(circles))
    plot_img = resized.copy()
    coords = circles[0,:]
    for i in coords:
        # draw the outer circle
        cv.circle(plot_img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(plot_img,(i[0],i[1]),2,(0,0,255),3)

    cv.imwrite('out/balls_detected.jpeg', plot_img)

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

    grid_img = resized.copy()
    cv.rectangle(grid_img, (min_x, min_y), (min_x + grid_size, min_y + grid_size),(255, 0, 0), 2)
    cv.imwrite('out/grid_detected.jpeg', grid_img)

    res = resized[min_y:min_y + grid_size, min_x:min_x + grid_size]
    cv.imwrite('out/grid_only.jpeg', res)
    return res


def process_image(img_file):
    grid_img = detect_grid(img_file)
    oh, ow, d = grid_img.shape
    img_gray = cv.cvtColor(grid_img, cv.COLOR_BGR2GRAY)
    perfect_rectangle_size = 1200
    scale = perfect_rectangle_size / oh
    nw, nh = ow * scale, oh * scale
    resized = cv.resize(img_gray, (math.ceil(nw), math.ceil(nh)), interpolation=cv.INTER_AREA)
    cv.imwrite('out/resized.jpeg', resized)
    elem_size = 150
    map = np.zeros((8, 8), dtype=np.dtype('U2'))
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
            templates_map = {'yellow.jpeg': 'YE', 'green.jpeg': 'GR', 'red.jpeg': 'RE', 'blue.jpeg': 'BL', 'brown.jpeg': 'BR', 'violet.jpeg': 'VI', 'skull.jpeg': 'SK', 'rock_skull.jpeg': 'RS', 'block.jpeg': 'BK'}
            for template_name in ['yellow.jpeg', 'green.jpeg', 'red.jpeg', 'blue.jpeg', 'brown.jpeg', 'violet.jpeg', 'skull.jpeg', 'rock_skull.jpeg', 'block.jpeg']:
                # print('matching {}'.format(template_name))
                template = cv.imread('templates/{}'.format(template_name), cv.IMREAD_COLOR)
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

def process_image_old(img_file):
    # load image
    img_rgb = cv.imread(img_file)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    ow, oh = img_gray.shape[::-1]
    # resize image
    perfect_rectangle_size = 1200
    scale = perfect_rectangle_size / oh
    nw, nh = ow * scale, oh * scale
    resized = cv.resize(img_gray, (math.ceil(nw), math.ceil(nh)), interpolation=cv.INTER_AREA)
    cv.imwrite('scaled.png', resized)

    marked_image = resized.copy()

    all_matches = []
    threshold = {'yellow.jpeg': 0.9, 'green.jpeg': 0.9, 'red.jpeg': 0.85, 'blue.jpeg': 0.9, 'brown.jpeg': 0.9, 'violet.jpeg': 0.85, 'skull.jpeg': 0.8, 'rock_skull.jpeg': 0.8}
    for template in ['yellow.jpeg', 'green.jpeg', 'red.jpeg', 'blue.jpeg', 'brown.jpeg', 'violet.jpeg', 'skull.jpeg', 'rock_skull.jpeg']:
    # for template in ['rock_skull.jpeg']:
        print("matching {}".format(template), cv.IMREAD_COLOR)
        matches = find_template_matches(resized, template, threshold[template])
        add_matches(marked_image, matches)
        all_matches += matches
    print(len(all_matches))

    left = min(all_matches, key=lambda x: x[0])[0]
    right = max(all_matches, key=lambda x: x[0])[0]
    top = min(all_matches, key=lambda x: x[1])[1]
    bottom = max(all_matches, key=lambda x: x[1])[1]

    for i in all_matches:
        i[0] = i[0] - left
        i[1] = i[1] - top

    cell_size = (right - left) / 7
    print('cell size: {}'.format(cell_size))

    all_matches.sort(key=lambda x: x[0])

    map = np.zeros((8, 8), dtype=np.dtype('U2'))
    for i in range(8):
        for j in range(8):
            map[i][j] = 'un'

    for m in all_matches:
        x = round(m[0] / cell_size)
        y = round(m[1] / cell_size)
        # print(x, y)
        map[y][x] = m[3]

    for i in range(8):
        res = ''
        for j in range(8):
            res += '{} '.format(map[i][j])
        print(res)
    return map


def scale_template(filename):
    img_rgb = cv.imread(filename)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    ow, oh = img_gray.shape[::-1]
    # resize image
    perfect_rectangle_size = 141
    scale = perfect_rectangle_size / oh
    nw, nh = ow * scale, oh * scale
    resized = cv.resize(img_gray, (math.ceil(nw), math.ceil(nh)), interpolation=cv.INTER_AREA)
    cv.imwrite('scaled.jpeg', resized)


process_image("input/gow_with_blocks.png")
# scale_template("templates/block.png")
