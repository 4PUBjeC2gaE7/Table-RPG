import numpy as np
import cv2
from ScreenRemap import runCalibration
from pynput import keyboard as kb

myLoop = True
sW = 1280
sH = 720

myAuras = { 'One':None, 'Two':None, 'Water': None, 'Fire':None}
for select in myAuras.keys():
    imOver = None
    imAlpha = None
    try:
        imOver = cv2.imread(f"./{select} Aura.png").astype(float)
    except:
        print(f'Failed to read "{select} Aura.png"')
    try:
        imAlpha = cv2.imread(f"./{select} Aura - Alpha.png")
        imAlpha = imAlpha / imAlpha.max()
    except:
        print(f'Failed to read "{select} Aura.png"')
    if imOver is not None and imAlpha is not None:
        myAuras[select] = (imOver, imAlpha)

def keyPress(key):
    global myLoop
    match key:
        case kb.Key.space:
            myLoop = False
        case kb.Key.esc:
            myLoop = False

def drawAura(imDst, coords, select):
    global myAuras
    imSrc = myAuras[select][0]
    imAlpha = myAuras[select][1]

    # calculate positions
    Oy, Ox, _ = imSrc.shape
    top = int (coords[1] - (Oy / 2))
    bottom = top + Oy
    left = int (coords[0] - (Ox / 2))
    right = left + Ox

    print(f'\ttop:{top}\tbottom:{bottom}\n\tleft:{left}\tright:{right}')
    # too far up
    if top < 0:
        Iy_start = 0
        Oy_start = abs(top)
    else:
        Iy_start = top
        Oy_start = 0
    # too far down
    if bottom > imDst.shape[0]:
        Iy_stop = imDst.shape[0]
        Oy_stop = Oy - (bottom - Iy_stop)
    else:
        Iy_stop = bottom
        Oy_stop = Oy
    # too far left
    if left < 0:
        Ix_start = 0
        Ox_start = abs(left)
    else:
        Ix_start = left
        Ox_start = 0
    # too far right
    if right > imDst.shape[1]:  
        Ix_stop = imDst.shape[1]
        Ox_stop = Ox - (right - Ix_stop)
    else:
        Ix_stop = right
        Ox_stop = Ox

    imSubOver = imSrc[Oy_start:Oy_stop, Ox_start:Ox_stop]
    imSub = imDst[Iy_start:Iy_stop, Ix_start:Ix_stop].astype(float)
    if imAlpha is not None:
        imSubOver *= imAlpha[Oy_start:Oy_stop, Ox_start:Ox_stop]
        imSub *= 1 - imAlpha[Oy_start:Oy_stop, Ox_start:Ox_stop]
    imSub = cv2.add(imSubOver, imSub)
    imDst[Iy_start:Iy_stop, Ix_start:Ix_stop] = imSub.astype(np.uint8)

    return imDst

if __name__ == '__main__':
    pts, frmMask = runCalibration()
    winMain = 'Reference Screen'
    winLive = 'Live View'

    H, ret = cv2.findHomography(np.array(pts), np.array([[0, sH],[0, 0],[sW, 0],[sW, sH]]))

    # define blob parameters, search
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.filterByArea = True
    blobParams.filterByColor = False
    blobParams.filterByInertia = False
    blobParams.filterByCircularity = False
    blobParams.filterByConvexity = False
    blobParams.minArea = 2000
    blobParams.maxArea = 50000
    myBlob = cv2.SimpleBlobDetector_create(blobParams)

    # generate base image
    vid = cv2.VideoCapture(0)
    imSrc = cv2.imread('background.png')
    cv2.imshow(winMain, imSrc)
    cv2.waitKey(250)
    ret, frmBase = vid.read()

    kbListen = kb.Listener(on_press=keyPress)
    kbListen.start()

    # grab frame; compare with ouput image
    while myLoop:
        imOut = imSrc.copy()
        ret, frmCurr = vid.read()
        if ret:
            # Grab Frame
            frmCurr = cv2.warpPerspective(frmCurr, H, (sW, sH))
            frmDiff = imOut.copy()
            frmCurr = cv2.cvtColor(frmCurr, cv2.COLOR_BGR2GRAY)
            frmDiff = cv2.cvtColor(frmDiff, cv2.COLOR_BGR2GRAY)
            frmDiff = cv2.GaussianBlur(frmDiff, (17, 17), 0)
            cv2.subtract(frmDiff, frmCurr, frmDiff)
            frmDiff = cv2.GaussianBlur(frmDiff, (19, 19), 0)

            # generate image threshold
            frmThres = frmDiff.copy()
            frmThres = cv2.GaussianBlur(frmThres, (21, 21), 0)
            cv2.multiply(frmThres, 255/frmThres.max(), frmThres)
            frmThres = cv2.erode(frmThres,np.ones(19, np.uint8))
            cv2.medianBlur(frmThres, 23, frmThres)
            _, frmThres = cv2.threshold(frmThres, 35, 255, cv2.THRESH_BINARY)
            frmThres = frmThres.max() - frmThres
            cv2.rectangle(frmThres, (0,0), (sW-1,sH-1), 255, 8)
            keyPoints = myBlob.detect(frmThres)
            myPts = cv2.KeyPoint_convert(keyPoints)

            try:
                print(f'Points found: {myPts.shape[0]}')
            except:
                pass
            
            imOut = imSrc.copy()
            for idx, pt in enumerate(myPts):
                print(f'Point {idx:-3d}: {pt}')
                imOut = drawAura(imOut, pt, list(myAuras.keys())[idx % len(myAuras.keys())])

        cv2.imshow(winMain, imOut)
        cv2.waitKey(1000)

    kbListen.stop()
