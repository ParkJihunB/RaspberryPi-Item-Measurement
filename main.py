import cv2
import numpy as np
import laser

frameWidth = 800#640
frameHeight = 600#480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
standW = 5
standH = 5
standD = 19

def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
#empty is a function whenever there are change in the value
cv2.createTrackbar("Threshold1","Parameters", 70,255,empty) 
cv2.createTrackbar("Threshold2", "Parameters",70,255,empty)
cv2.createTrackbar("Area","Parameters",1000,30000,empty)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvaliable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvaliable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] =  cv2.resize(imgArray[x][y], (0,0), None, scale,scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height,width,3),np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]),None,scale,scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return  ver

#@img: input img(find the contour from)
#@imgContour: write down detected contour
def getContours(img, imgContour):
    #Retrieval method
        #EXTERNAL: retrieve oonly extreme outer corner
        #CHIN_APROX_'NONE': Get all of the approx points('SIMPLE' get less points)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #display contours on our img(this is why we have imgContour param)
    #cv2.drawContours(imgContour,contours,-1,(255,0,255), 3)#width = 7
    
    #detect area of each contour and base on the area
    #remove all the contours that not interested in
    #so reduce the noise
    finalCountours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area","Parameters")
        if area<areaMin: continue#filter contours
        cv2.drawContours(imgContour, cnt, -1,(255,0,255),2)#width = 3
        #it retrn value of length of the parameter
        peri = cv2.arcLength(cnt, True)#True: contour is closed
        #using parameter to approxtimate what type of shape is this.
        #(contour/resolution/this is closed contour
        #apporx=array have certain amount of points
        approx = cv2.approxPolyDP(cnt,0.02*peri, True)
        #print(len(approx))#print number of point(able to assume shape)
        #create bounding box because the points might not be always in sqaure form..
        #or other many reasons
        x,y,w,h = cv2.boundingRect(approx)
        #rectangle(bounding box) will drawn on the image.
        #x and y as inital point
        #x+w,x+h is final point of bounding box  
        finalCountours.append([len(approx),area,approx,cnt,[x,y,w,h]])
        
    finalCountours = sorted(finalCountours, key = lambda x:x[1], reverse=True)
    if len(finalCountours) ==2:
        return [finalCountours[0],finalCountours[1]] #return biggest and next biggest
    else: return 0

def applyStandard(con):
    width = con[4][2]
    height = con[4][3]
    scaleW = standW/width
    scaleH = standH/height
    return [scaleW,scaleH]

def scaleWH(con,scale):
    width = con[4][2] * scale[0]
    height = con[4][3] * scale[1]
    area = width * height
    con[1] = area
    return [int(width),int(height)]

def drawCon(con,imgContour, conWH):
    font_thick = 2
    font_size = 1
    font_color = (200,0,200)
    x = con[4][0]
    y = con[4][1]
    w = con[4][2]
    h = con[4][3]
    cv2.rectangle(imgContour, (x, y), (x+w, y+h),(0,255,0),3);
    cv2.putText(imgContour, "Points: "+str(con[0]), (x+w//2,y+h//2), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)
    cv2.putText(imgContour, "Area: "+str(int(con[1])), (x+20,y+35), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)
    cv2.putText(imgContour, "Width: "+str(conWH[0]), (x,y), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)
    cv2.putText(imgContour, "Height: "+str(conWH[1]), (x-w//3,y+h//2), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)
    
objResult = [0,0,0]
resultCount = 0

while True:
    success, img = cap.read()
    #blur kernel by (7,7)
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    #Get the trackbarpos of threshold in parameter bar...
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5,5))
    #better outer edges img
    imgDil = cv2.dilate(imgCanny, kernel, iterations = 1)#delclare dilation
    
    imgContour = img.copy()
    contours = getContours(imgDil, imgContour)

    if contours !=0 :
        #give second small obj(=standard) and get scale rate
        con = contours[1]
        scale = applyStandard(con)
        conWH = scaleWH(con,scale)
        drawCon(con, imgContour, conWH)
        
        con = contours[0]
        conWH = scaleWH(con, scale)
        drawCon(con, imgContour, conWH)
        depth = laser.get_depth(standD)
        if objResult[0] == 0: objResult = [conWH[0],conWH[1],depth]
        else:
            objResult[0] = int((objResult[0]+conWH[0])*0.5)
            objResult[1] = int((objResult[1]+conWH[1])*0.5)
            objResult[2] = int((objResult[2]+depth)*0.5)
    
    if resultCount > 10:
        print(objResult)
        resultCount = 0
    resultCount += 1

    #imgStack = stackImages(0.8,([img,imgGray,imgCanny],
                                #[imgDil, imgContour, imgContour])) #stack images to array
    imgStack = stackImages(0.8,([img,imgDil,imgContour]))
    
    cv2.imshow("Result",imgStack)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break