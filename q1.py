import numpy as np
import cv2
import random


def minCut(imgCut1, imgCut2):
    df = 2
    ssd = np.sum((imgCut1 - imgCut2) ** 2, axis=2, dtype='uint32')
    hei, wid = ssd.shape
    minTotalE = float('inf')
    bestCut = []
    for j in range(wid):
        totalE = ssd[0, j]
        pixelsList = []
        jCur = j
        pixelsList.append(jCur)
        for i in range(hei - 1):
            window = ssd[i + 1, max(0, (jCur - df)):min(wid, (jCur + df + 1))]
            totalE = totalE + np.min(window)
            dj = np.argmin(window) - min(jCur, df)
            jCur = jCur + dj
            pixelsList.append(jCur)
        if (totalE < minTotalE):
            minTotalE = totalE
            bestCut = pixelsList
    return np.array(bestCut)


def merge(img1, img2, bandWidth):
    hei, wid, chan = img1.shape
    imgCut1 = img1[:, (wid - bandWidth):(wid), :]
    imgCut2 = img2[:, (0):(bandWidth), :]
    bestCut2 = minCut(imgCut1, imgCut2)
    bestCut1 = bestCut2 + wid - bandWidth
    imgMerged = np.zeros((hei, 2 * wid - bandWidth, 3))
    for i in range(hei):
        imgMerged[i, :bestCut1[i], :] = img1[i, :bestCut1[i], :]
        imgMerged[i, bestCut1[i]:, :] = img2[i, bestCut2[i]:, :]
        imgMerged[i, bestCut1[i], :] = img1[i, bestCut1[i], :] / 2 + img2[i, bestCut2[i], :] / 2
    # bluring
    for i in range(hei):
        imgMerged[max((i - 2), 0):min((i + 3), hei), (bestCut1[i] - 2): (bestCut1[i] + 3), :] = cv2.blur(
            imgMerged[max((i - 2), 0):min((i + 3), hei), (bestCut1[i] - 2): (bestCut1[i] + 3), :], (3, 3))
    return imgMerged


def initiate(img, hei, wid):
    height, width, channels = img.shape
    i = random.randint(0, height - hei)
    j = random.randint(0, width - wid)
    return img[i:i + hei, j:j + wid, :]


def findBestPatch(img, imgTemplate, bandWidth):
    heiPtch, widPtch, chanPtch = imgTemplate.shape
    imgTmp = cv2.matchTemplate(img, imgTemplate[:, widPtch - bandWidth:widPtch, :], cv2.TM_CCOEFF_NORMED)
    imgTmp = imgTmp[:, 0:imgTmp.shape[1] + bandWidth - widPtch]
    # i, j = np.where(imgTmp - np.max(imgTmp) == 0)
    i, j = np.where(imgTmp > np.max(imgTmp) - 0.1)
    index = random.randint(0, len(i) - 1)
    templateMatched = img[i[index]:i[index] + heiPtch, j[index]:j[index] + widPtch, :]
    return templateMatched

# def findBestPatch2(img, imgTemplateL, imgTemplateU, bandWidth):
#     heiPtch, widPtch, chanPtch = imgTemplateL.shape
#     imgTmpL = cv2.matchTemplate(img, imgTemplateL[:, widPtch - bandWidth:widPtch, :], cv2.TM_CCOEFF_NORMED)
#     imgTmpL = imgTmpL[:, 0:imgTmpL.shape[1] + bandWidth - widPtch]
#     imgTmpU = cv2.matchTemplate(img, imgTemplateU[heiPtch - bandWidth:heiPtch, :, :], cv2.TM_CCOEFF_NORMED)
#     imgTmpU = imgTmpU[0:imgTmpU.shape[0] + bandWidth - heiPtch, :]
#     imgTmp = imgTmpL / 2 + imgTmpU / 2
#     # i, j = np.where(imgTmp - np.max(imgTmp) == 0)
#     i, j = np.where(imgTmp > np.max(imgTmp) - 0.1)
#     index = random.randint(0, len(i) - 1)
#     templateMatched = img[i[index]:i[index] + heiPtch, j[index]:j[index] + widPtch, :]
#     return templateMatched

def findBestPatch2(img, imgTemplateL, imgTemplateU, bandWidth):
    heiPtch, widPtch, chanPtch = imgTemplateL.shape
    imgTmpL = cv2.matchTemplate(img, imgTemplateL[:, widPtch - bandWidth:widPtch, :], cv2.TM_CCOEFF_NORMED)
    imgTmpL = imgTmpL[:, 0:imgTmpL.shape[1] + bandWidth - widPtch]
    imgTmpU = cv2.matchTemplate(img, imgTemplateU[heiPtch - bandWidth:heiPtch, bandWidth:, :], cv2.TM_CCOEFF_NORMED)
    imgTmpU = imgTmpU[0:imgTmpU.shape[0] + bandWidth - heiPtch, bandWidth:]
    imgTmp = imgTmpL / 2 + imgTmpU / 2
    # i, j = np.where(imgTmp - np.max(imgTmp) == 0)
    i, j = np.where(imgTmp > np.max(imgTmp) - 0.1)
    index = random.randint(0, len(i) - 1)
    templateMatched = img[i[index]:i[index] + heiPtch, j[index]:j[index] + widPtch, :]
    return templateMatched


def synthesis(img, heiPtch, widPtch, heiTrg, widTrg, bandWidth):
    heiCnt, widCnt = int(heiTrg / heiPtch), int(widTrg / widPtch)
    heiPtch, widPtch = heiPtch + bandWidth, widPtch + bandWidth
    heiTrg, widTrg = heiTrg + bandWidth, widTrg + bandWidth
    imgSyn = np.zeros((heiTrg, widTrg, 3), dtype='uint8')
    # first cell
    ptchF = initiate(img, heiPtch, widPtch)
    imgSyn[0:heiPtch, 0:widPtch, :] = ptchF
    # fist row cells
    for i in range(1, widCnt):
        ptchL = imgSyn[0:heiPtch, (i - 1) * (widPtch - bandWidth):(i - 1) * (widPtch - bandWidth) + widPtch, :]
        ptchR = findBestPatch(img, ptchL, bandWidth)
        imgSyn[0:heiPtch, (i - 1) * (widPtch - bandWidth):(i - 1) * (widPtch - bandWidth) + 2 * widPtch - bandWidth,
        :] = merge(ptchL, ptchR,
                   bandWidth)
    # first column cells
    for j in range(1, heiCnt):
        ptchU = imgSyn[(j - 1) * (heiPtch - bandWidth):(j - 1) * (heiPtch - bandWidth) + heiPtch, 0:widPtch, :]
        ptchU = np.rot90(ptchU)
        ptchD = findBestPatch(np.rot90(img), ptchU, bandWidth)
        imgSyn[(j - 1) * (heiPtch - bandWidth):(j - 1) * (heiPtch - bandWidth) + 2 * heiPtch - bandWidth, 0:widPtch,
        :] = np.rot90(
            merge(ptchU, ptchD, bandWidth), -1)
    # other cells
    for j in range(1, heiCnt):
        print("\r", str(int(j / (heiCnt - 1) * 100)), "%", end="")
        for i in range(1, widCnt):
            ptchL = imgSyn[j * (heiPtch - bandWidth):j * (heiPtch - bandWidth) + heiPtch,
                    (i - 1) * (widPtch - bandWidth):(i - 1) * (widPtch - bandWidth) + widPtch, :].copy()
            ptchU = imgSyn[(j - 1) * (heiPtch - bandWidth):(j - 1) * (heiPtch - bandWidth) + heiPtch,
                    i * (widPtch - bandWidth):i * (widPtch - bandWidth) + widPtch, :].copy()
            ptchM = findBestPatch2(img, ptchL, ptchU, bandWidth)

            imgSyn[j * (heiPtch - bandWidth):j * (heiPtch - bandWidth) + heiPtch,
            (i - 1) * (widPtch - bandWidth):(i) * (widPtch - bandWidth) + widPtch, :] = merge(ptchL, ptchM, bandWidth)

            ptchM = imgSyn[j * (heiPtch - bandWidth):j * (heiPtch - bandWidth) + heiPtch,
                    i * (widPtch - bandWidth):i * (widPtch - bandWidth) + widPtch, :]
            ptchM = np.rot90(ptchM)
            ptchU = np.rot90(ptchU)
            imgSyn[(j - 1) * (heiPtch - bandWidth):(j) * (heiPtch - bandWidth) + heiPtch,
            i * (widPtch - bandWidth):i * (widPtch - bandWidth) + widPtch, :] = np.rot90(merge(ptchU, ptchM, bandWidth),
                                                                                         -1)

    imgSyn = imgSyn[0:heiTrg - bandWidth, 0:widTrg - bandWidth, :]
    return imgSyn


bandWidth = 35
I = cv2.imread("texture1.jpg")
heiPtch, widPtch = 125, 125
heiTrg, widTrg = 2500, 2500
cv2.imwrite("res1.jpg", synthesis(I, heiPtch, widPtch, heiTrg, widTrg, bandWidth))
