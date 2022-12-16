import numpy as np
import matplotlib.pyplot as plt
import cv2


def getGradient(gray, x = 0, y = 0, useGradient = True):
    if useGradient:
        grad = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=x, dy=y, ksize=3)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        if maxVal - minVal > 0:
            grad = (255 * ((grad - minVal) / float(maxVal - minVal))).astype("uint8")
        else:
            grad  = np.zeros(gray.shape, dtype = "uint8")

    else:
        grad = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    return grad



def show_img(img, w = 20, h = 20):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(img, cmap='gray')
    plt.show()

datasetInfo = []
imageCol = []


with open('input_seg.txt') as input_coords:
    lines = input_coords.readlines()
    for idx in range(0, len(lines), 2):
        name, coords = lines[idx: idx + 2]
        datasetInfo.append([name.strip(), coords.split()])

for item in datasetInfo:
    path = '{}'.format(item[0])
    x1, y1, x2, y2 = map(int, item[1])
    image = cv2.imread(path)[x1:x2, y1:y2]
    imageCol.append(image)

def concat_hor(imgs, color = (0,255,0)):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[0])
        s += img.shape[1]+2*bs


    image = np.zeros((m+2*bs, s, 3))

    x = 0
    for img in imgs:
        if len(img.shape) == 3:
            imgg = cv2.copyMakeBorder(img.copy(), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=color)
            image[0:imgg.shape[0], x:x+imgg.shape[1], :] = imgg
        else:
            imgg = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=color)
            image[0:imgg.shape[0], x:x+imgg.shape[1], :] = imgg
        x += img.shape[1]+2*bs

    return np.asarray(image, dtype = np.uint8)


def concat_ver(imgs):
    m = 0
    s = 0
    bs = 1
    for img in imgs:
        m = max(m, img.shape[1])
        s += img.shape[0]+2*bs


    image = np.zeros((s, m+2*bs, 3))

    y = 0
    for img in imgs:
        if len(img.shape) == 3:
            imgg = cv2.copyMakeBorder(img.copy(), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=(0, 255, 0))
            image[y:y+imgg.shape[0], 0:imgg.shape[1], :] = imgg
        else:
            imgg = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=(0, 255, 0))
            image[y:y+imgg.shape[0], 0:imgg.shape[1], :] = imgg
        y += img.shape[0]+2*bs

    return np.asarray(image, dtype = np.uint8)


def getDrawProjectionVer(lp, verp):
    verp2 = verp.astype(int)
    w = np.max(verp2) + 5
    graphicVer = np.zeros((lp.shape[0], w), dtype = "uint8")

    for i in range(len(verp2)):
        graphicVer[i, 0:verp2[i]] = 255

    return cv2.cvtColor(graphicVer, cv2.COLOR_GRAY2BGR)


def getDrawProjectionHor(lp, horp):
    horp2 = horp.astype(int)
    h = int(np.max(horp2) + 5)
    graphicHor = np.zeros((h, lp.shape[1]), dtype = "uint8")

    for i in range(len(horp2)):
        graphicHor[int(graphicHor.shape[0]-horp2[i]):graphicHor.shape[0], i] = 255

    return cv2.cvtColor(graphicHor, cv2.COLOR_GRAY2BGR)


def findb0(verpConvolved, ybm, c):
    for i in range(ybm,-1,-1):
        if verpConvolved[i] <= c:
            return i
    return 0


def findb1(verpConvolved, ybm, c):
    for i in range(ybm,len(verpConvolved)):
        if verpConvolved[i] <= c:
            return i
    return len(verpConvolved)


def getHOGFeatures(img):
    win_size = (48, 48)
    nbins = 4  # number of orientation bins
    cell = (5,5)  # h x w in pixels

    hog_temp = cv2.HOGDescriptor(_winSize=(win_size[0], win_size[1]),_blockSize=(win_size[0], win_size[1]),
                                _blockStride=(cell[1], cell[0]),
                                _cellSize=(cell[1], cell[0]),
                                _nbins=nbins, _histogramNormType = 0, _gammaCorrection = True)


def horizontal_segmentation(img):
#     show_img(img)
    img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#     show_img(img_bin)
    horVals = np.sum(img_bin, axis=0) / 255
    horProj = getDrawProjectionHor(img_bin, horVals)
#     show_img(horProj)
    kernel = np.array([-1, 3, 7, 3, -1]) / 11
    horConv = np.convolve(horVals, kernel, mode='same')
    horProjConv = getDrawProjectionHor(img_bin, horConv)
#     show_img(horProjConv)
    threshVal = np.min(horConv)
    print(threshVal)
    medianWidth = int(img.shape[0] * 0.3)
    bandP1rangesH = []
    peaksH = []
    c1, c2 = 0.09, 0.12
    while np.max(horConv) > 0:
        ybm = np.argmax(horConv)
        yb0 = findb0(horConv, ybm, c1 * horConv[ybm])
        yb1 = findb1(horConv, ybm, c2 * horConv[ybm])
        if yb1 - yb0 > medianWidth:
            bandP1rangesH.append((yb0, yb1))
            peaksH.append((int(horConv[ybm]), ybm))
        horConv[yb0:yb1] = 0
    wordBand = img.copy()
    bandRanges = sorted([item for t in bandP1rangesH for item in t])
    bandRanges_ = [0]
    for band in bandRanges:
        if band - bandRanges_[-1] <= 0.7 * medianWidth:
            bandRanges_[-1] = (band + bandRanges_[-1]) // 2
        elif band - bandRanges_[-1] <= 1.4 * medianWidth:
            bandRanges_[-1] = bandRanges_[-1]
        else:
            bandRanges_.append(band)
    for band in bandRanges_:
        wordBand = cv2.line(wordBand, (band, 0), (band, wordBand.shape[0]), (0, 255, 0), 1)
    show_img(wordBand)
    bandRanges_ = [0] + bandRanges_ + [img.shape[1]]
    return bandRanges_


horizontal_segmentation(imageCol[0])


for image in imageCol:
    horizontal_segmentation(image)