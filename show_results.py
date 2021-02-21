import cv2 as cv
import os
import numpy as np

test_data_path = 'training_samples/'
imgnames = [f for f in os.listdir(test_data_path) if os.path.isfile(os.path.join(test_data_path, f)) and os.path.splitext(f)[1] != '.txt']
assert len(imgnames)

winname = 'result'
cv.namedWindow(winname, cv.WINDOW_KEEPRATIO)

for imgname in imgnames:
    # read test image
    imgpath = os.path.join(test_data_path, imgname)
    img = cv.imread(imgpath, cv.IMREAD_COLOR)
    if img is None:
        print('%s is not an image! Skipping...' % imgpath)
        continue
    # read corresponding detection result
    outpath = 'outputs/res_' + os.path.splitext(imgname)[0] + '.txt'
    if not os.path.isfile(outpath):
        print('Could not find output file in %s ! Skipping...' % outpath)
        continue
    with open(outpath, 'r') as f:
        lines = f.readlines()
    # draw detection results
    res = np.copy(img)
    for line in lines:
        line = line[0:-1]   # strip newline
        output = line.split(',')
        if len(output) != 9:
            print('Invalid output in %s: %s' % (outpath, line))
            continue
        box = np.array([int(b) for b in output[0:8]], dtype=np.int32).reshape((4, 2))
        recognition_result = output[8]
        # Draw bounding box
        cv.polylines(res, box.reshape((1, 4, 2)), isClosed=True, color=(255, 255, 0))
        # Draw recognition results area
        text_area = box.copy()
        text_area[2, 1] = text_area[1, 1]
        text_area[3, 1] = text_area[0, 1]
        text_area[0, 1] = text_area[0, 1] - 15
        text_area[1, 1] = text_area[1, 1] - 15
        cv.fillPoly(res, text_area.reshape((1, 4, 2)), color=(255, 255, 0))
        cv.putText(res, recognition_result, (box[0, 0], box[0, 1]), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), lineType=cv.LINE_AA)
    # display result
    cv.imshow(winname, res)
    ch = cv.waitKey() & 0xFF
    if ch == 27:    # ESC
        break
    if chr(ch).lower() == 's':
        resname = 'outputs/' + imgname
        if cv.imwrite(resname, res):
            print('Wrote out %s' % resname)
        else:
            print('Could not write to %s' % resname)
