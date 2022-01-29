import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 100  # frame reduction
smoothening = 5 # this will add delay
##########################

pTime = 0
plocX, plocY = 0, 0 # privious location of x and y
clocX, clocY = 0, 0 # current location of x and y

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. get the tip of the index and middle finger
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:] # will give me coordinate
        x2, y2 = lmList[12][1:]
        # print(x1,y1,x2,y2)

        # 3. check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2) # this will reduce frame
        # 4. only first finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. convert the coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # 6. smoothen the values
            clocX = plocX+(x3-plocX) /smoothening
            clocY = plocY+(y3-plocY) /smoothening

            # 7. move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED) # this will show me circle
            plocX,plocY = clocX, clocY

        # 8. check if we are in clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10. click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15,
                           (0,255,0),cv2.FILLED)
                autopy.mouse.click()



    # 11. frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    # 12. Display
    image = cv2.flip(img,1)
    cv2.imshow("Image", image)
    cv2.waitKey(1)
