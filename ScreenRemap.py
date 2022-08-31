import cv2
import time
import math
import numpy as np
import screeninfo
from pynput import keyboard as kb

winMain = 'Reference Screen'
winLive = 'Live View'
myThres = 130
Abort = False
Loopy = True
sW = 1280
sH = 720

def keyPress(key):
    global Loopy, Abort
    match key:
        case kb.Key.space:
            Loopy = False
        case kb.Key.esc:
            Loopy = False
            Abort = True

def keyPress_wThres(key):
    global myThres, Loopy, Abort
    match key:
        case kb.Key.up:
            myThres += 5
        case kb.Key.down:
            myThres -= 5
        case kb.Key.space:
            Loopy = False
        case kb.Key.esc:
            Loopy = False
            Abort = True

    print(f'\tThreshold: {myThres:3d}')

def calculateLineAverage(lineSet):
    avg_r = 0
    avg_t = 0
    for elem in lineSet:
        avg_r += elem[0]
        avg_t += elem[1]
    avg_r /= len(lineSet)
    avg_t /= len(lineSet)
    return (avg_r, avg_t)

def parseLine(polarLine):
    if polarLine[1] == 0:
        slope = np.inf
        intercept = polarLine[0]
    else:
        slope = - np.cos(polarLine[1]) / np.sin(polarLine[1])
        intercept = polarLine[0] / np.sin(polarLine[1])
    return(slope, intercept)

def runCalibration():
    global myThres, Loopy, Abort, sW, sH
    # start camera
    vid = cv2.VideoCapture(0)
    ret, frmCurr = vid.read()
    if not ret:
        raise Exception("Camera not found") 

    # start keyboard listener
    kbListen = kb.Listener(on_press=keyPress_wThres)
    if __name__ == '__main__':
        print("""
+-------------+
| User Inputs |
+-------------+
arrow up/down:    Threshold
escape:           Quit

""")
    kbListen.start()

    # query screen info
    try:
        screen = screeninfo.get_monitors()[1]
        sW = screen.width
        sH = screen.height
        print(f'Projector found: {sW:3d}x{sH:d}\n')
    except:
        print('Screen not found\n')
        screen = None

    # create reference image
    refImage = np.zeros((sH,sW,3), dtype=np.float32)
    # draw cross
    cv2.line(refImage, (0,0), (sW,sH),(0,0.2,0.7), 13, cv2.LINE_AA)
    cv2.line(refImage, (sW,0), (0,sH),(0,0.2,0.7), 13, cv2.LINE_AA)
    # draw grid
    for i in range(10):
        refImage[:, (sW // 10 + 1) * i] = [1,0,0]
        refImage[(sH // 10 + 1) * i, :] = [1,0,0]
    # draw border
    refImage[:,:10] = [0,1,0]
    refImage[:,sW-10:] = [0,1,0]
    refImage[:10,:] = [0,1,0]
    refImage[sH-10:,:] = [0,1,0]
    # draw text
    cv2.putText(refImage,'Center Frame In Camera',
                (sW // 4, sH // 4),
                cv2.FONT_HERSHEY_PLAIN,
                3, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(refImage,'Press <SPACE> When Ready',
                (sW // 4, sH * 3 // 4),
                cv2.FONT_HERSHEY_PLAIN,
                3, (255,255,255), 2, cv2.LINE_AA)
    
    # move image to projector screen
    cv2.namedWindow(winMain, cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(winMain, refImage)
    if screen is not None:
        cv2.moveWindow(winMain, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(winMain, cv2.WND_PROP_FULLSCREEN,
                                       cv2.WINDOW_FULLSCREEN)
    else:
        cv2.moveWindow(winMain, 200, 100)
        cv2.setWindowProperty(winMain, cv2.WND_PROP_FULLSCREEN,
                                       cv2.WINDOW_NORMAL)

    # allow user to align image
    while Loopy:

        # capture the next frame
        ret, frmCurr = vid.read()

        # run Canny edge detect
        frmCann = cv2.Canny(frmCurr, 50, 200, None, 3)
        frmGrey = cv2.cvtColor(frmCann, cv2.COLOR_GRAY2BGR)
        frmCurr = cv2.addWeighted(frmCurr, 0.7, frmGrey, 0.3, 0)

        # run Hough line detect
        lines = cv2.HoughLinesP(frmCann, 1, np.pi/180, myThres, None, 50, 10)

        # plot lines on screen
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(frmCurr, (l[0], l[1]), (l[2], l[3]), (0,55,110), 1, cv2.LINE_AA)
            
        cv2.imshow(winLive, frmCurr)

        # check for user input
        cv2.waitKey(100)

    kbListen.stop()

    # scan for image border
    if not Abort:
        # capture base image
        refImage[:,:] = [0,0,0]
        cv2.imshow(winMain, refImage)
        cv2.waitKey(500)
        ret, frmBase = vid.read()
        cv2.imshow(winLive, frmBase)
        cv2.waitKey(200)

        # capture test images
        frmAcc = frmBase * 0
        for testColor in ([0,0,1], [0,1,0], [1,0,0]):
            refImage[:,:] = testColor
            cv2.imshow(winMain, refImage)
            cv2.waitKey(300)
            ret, frmCurr = vid.read()
            frmAcc = cv2.add(frmAcc, cv2.subtract(frmCurr, frmBase))
            cv2.imshow(winLive, frmAcc)
            cv2.waitKey(300)

        _, frmAcc = cv2.threshold(frmAcc, 100, 255, cv2.THRESH_BINARY)
        frmMask = cv2.erode(frmAcc, np.ones(7, np.uint8))
        frmCann = cv2.Canny(frmMask, 50, 200, None, 3)

        lines = []
        catLines = []
        while len(lines) < 4 or len(catLines) < 4:
            frmCurr = frmBase
            # run Hough line detect
            lines = cv2.HoughLines(frmCann, 1, np.pi/180, myThres)
            # categorize lines
            for i, myLine in enumerate(lines):
                r, theta = myLine[0]
                isMatch = False
                # initialize if empty
                if not catLines:
                    pass
                else:
                    for cat in catLines:
                        avgR, avgTheta = calculateLineAverage(cat)
                        if (abs(r - avgR) / avgR) < 0.2 and abs(theta - avgTheta) < (np.pi / 6):
                            cat.append((r, theta))
                            isMatch = True
                            break
                if not isMatch:
                    catLines.append([])
                    catLines[-1].append((r, theta))
            # check number of lines; adjust accordingly
            if len(catLines) < 4:
                myThres -= 1
            
        myColors = [(0,0,255),(0,255,255),(0,255,0),(255,220,0),(250,120,0),(255,0,0),(220,0,220),(0,128,255)]
        for i, cat in enumerate(catLines):
            # plot lines
            for myLine in cat:
                (m,b) = parseLine(myLine)
                if np.isinf(m):
                    pt1 = (int (b), 0)
                    pt2 = (int (b), 1000)
                else:
                    pt1 = (0, int (b))
                    pt2 = (1000, int (m*1000 + b))
                cv2.line(frmCurr, pt1, pt2, myColors[i % len(myColors)], 2)

            # calculate line averages
            catLines[i] = parseLine(calculateLineAverage(cat))

            # display image
            cv2.imshow(winLive, frmCurr)
            cv2.waitKey(50)
        
        # list lines
        if __name__ == '__main__':
            print('\nLines:')
            for myLine in catLines:
                print(f'm: {myLine[0]:.2f}\tb: {myLine[1]:.2f}')

        # calculate line intersections
        xPoints = []
        for i in range(len(catLines)):
            for j in range(i,len(catLines)):
                if catLines[i][0] != catLines[j][0]:
                    # x = (b2 - b1 ) / (m1 - m2)
                    x = (catLines[j][1] - catLines[i][1]) / (catLines[i][0] - catLines[j][0])
                    y = catLines[i][0]*x + catLines[i][1]
                    if x > 0 and y > 0 and x < sW and y < sH:
                        xPoints.append((x,y))
                        xy = (int (x), int (y))
                        cv2.circle(frmBase, xy, 5, (0,0,0), 2)
                        cv2.putText(refImage,f'{xy[0]:d},{xy[1]:d}',xy,
                                    cv2.FONT_HERSHEY_PLAIN,
                                    3, (0,0,0), 2, cv2.LINE_AA)
        xPoints.sort()

        # list lines
        if __name__ == '__main__':
            print('\nPoints:')
            for xy in xPoints:
                print(f'x: {xy[0]:.2f}\ty: {xy[1]:.2f}')

        vid.release()
        return xPoints, frmMask

if __name__ == '__main__':
    pts, frmMask = runCalibration()
    frmMask = cv2.cvtColor(frmMask, cv2.COLOR_BGR2GRAY)
    H, ret = cv2.findHomography(np.array(pts), np.array([[0, sH],[0, 0],[sW, 0],[sW, sH]]))


    # generate base image
    vid = cv2.VideoCapture(0)
    imgSrc = cv2.imread('background.png')
    imgOut = imgSrc.copy()
    cv2.imshow(winMain, imgSrc)
    cv2.waitKey(250)
    ret, frmBase = vid.read()

    kbListen = kb.Listener(on_press=keyPress)
    kbListen.start()

    Loopy = True
    while Loopy:
        ret, frmCurr = vid.read()
        frmDiff = frmCurr.copy()
        cv2.subtract(frmBase, frmCurr, frmDiff)
        frmDiff = cv2.cvtColor(frmDiff, cv2.COLOR_BGR2GRAY)
        cv2.bitwise_and(frmMask,frmDiff,frmDiff)
        frmThres = cv2.dilate(frmDiff,np.ones(7, np.uint8))
        cv2.threshold(frmThres, 127, 255, cv2.THRESH_BINARY)
        frmThres = cv2.GaussianBlur(frmThres, (17, 17), 0)
        cv2.bitwise_and(frmMask,frmThres,frmThres)
        frmThres = np.array(frmThres, dtype = np.uint8)
        cv2.imshow('test',frmThres)
        # frmHeat = cv2.applyColorMap(frmThres, cv2.COLORMAP_JET)
        frmThres = cv2.cvtColor(frmThres, cv2.COLOR_GRAY2BGR)
        frmWarp = cv2.warpPerspective(frmThres, H, (sW,sH))
        cv2.add(imgSrc, frmWarp, imgOut)
        # cv2.imshow(winMain, frmWarp)
        cv2.imshow(winMain, imgOut)
        cv2.imshow(winLive, frmCurr)
        cv2.waitKey(100)

    # Clean-up
    vid.release()
    cv2.destroyAllWindows()
    kbListen.stop()