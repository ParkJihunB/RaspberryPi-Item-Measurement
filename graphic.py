import cv2
import numpy as np

class Graphic:
    def __init__(self, width, height):
        #카메라를 인식하고 크기를 세팅합니다
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,width)
        self.cap.set(4,height)
        #self.init_window(640,240)

    def init_window(self,width, height):
        cv2.namedWindow("Parameters")
        #파라미터를 조정할수 있는 트랙바를 띄울 창을 생성하고 사이즈를 조정합니다.
        cv2.resizeWindow("Parameters",width,height)
        cv2.createTrackbar("Threshold1", "Parameters", 70, 255, self.empty)
        cv2.createTrackbar("Threshold2", "Parameters", 70, 255, self.empty)
        cv2.createTrackbar("Area", "Parameters", 1000, 30000, self.empty)

    def empty(self, a): pass #파라미터 생성시 사용됨

######################### HELPER FUNCTIONS #########################
#한번에 여러개의 이미지를 보기 위해 이미지를 쌓는 함수
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_avaliable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_avaliable:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] =  cv2.resize(img_array[x][y], (0,0), None, scale,scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y],cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height,width,3),np.uint8)
        hor = [image_blank]*rows
        hor_con = [image_blank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0,rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0,0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]),None,scale,scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return  ver

#img: 인풋 이미지
#img_contour: 측정된 윤곽을 그려 리턴
#area_min: 측정할 대상의 최소 넓이
#con_num: 측정된 contour이 몇개일 때만 리턴할지
# 인풋 이미지에서 윤곽(contours)을 구해 인풋 이미지 위에 입혀 리턴
def get_contours(img, img_contour, area_min, con_num):
    #인풋 이미지에서 contour 찾기
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    final_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt) #측정된 contour에서 넓이를 구합니다
        #현재 측정된 contour의 넓이가 최소넓이보다 작을 경우 건너뜁니다
        #이 필터링으로 이미지에서 측정된 모든 contour중 불필요한(노이즈) 것을 뛰어넘습니다
            #만약 파라미터 창을 띄워 트랙바로 직접 최소 넓이를 조정하고 싶다면
            #Grapic 클래스의 컨스트럭터에서 init_window 함수를 활성화하고
            #해당 창 이름과 파라미터 이름을 사용해 트랙바 포지션을 값으로 가져올 수 있습니다
            #areaMin = cv2.getTrackbarPos("Area","Parameters")
        if area<area_min: continue
        #width = 2로 측정된 윤곽선을 그립니다
        cv2.drawContours(img_contour, cnt, -1,(255,0,255),2)
        #contour의 길이 정보를 받아옵니다
        peri = cv2.arcLength(cnt, True)
        #길이 정보로 contour가 어떤 종류의 모양인지(삼각형/사각형..) 대략적으로 측정합니다
        #approxPolyDP(측정할 contour/resolytion/contour가 닫힌 모양인지 여부)
        #approx:측정된 모양에 기반한 포인트들이 저장된 배열(사각형이면 4개의 포인트)을 리턴받음
        approx = cv2.approxPolyDP(cnt,0.02*peri, True)
        #포인트 배열에 기반해 contour를 감싸는 바운딩박스 정보를 받아옵니다
        #바운딩 박스를 이용하는 이유는 사각형이 아닌 모양의 사이즈를 측정하기 위해서입니다
        #x,y는 inital point (예: 바운딩박스의 왼쪽 위 포인트)
        #x+w, y+h는 final point(예: 바운딩박스의 오른쪽 아래 포인트)
        x,y,w,h = cv2.boundingRect(approx)
        #최종 결과에 contour 정보를 모두 저장합니다
        final_contours.append([len(approx),area,approx,cnt,[x,y,w,h]])
    #측정된 contour들을 모두 정렬합니다
    final_contours = sorted(final_contours, key = lambda x:x[1], reverse=True)
    result = []
    if len(final_contours) ==con_num: #측정된 contour이 con_num개인 경우에만 리턴
        for i in range(con_num):
            result.append(final_contours[i])
        return result #return biggest and next biggest
    else: return 0
    
#contour 정보를 img_contour에 그립니다
def draw_con(con,img_contour, conWH):
    font_thick = 2
    font_size = 1
    font_color = (200,0,200)
    #final_contours.append([len(approx),area,approx,cnt,[x,y,w,h]]) 를 참고하여
    #x,y,w,h 값을 알 수 있습니다
    x = con[4][0]
    y = con[4][1]
    w = con[4][2]
    h = con[4][3]
    #x,y 값으로 바운딩박스를 그립니다
    cv2.rectangle(img_contour, (x, y), (x+w, y+h),(0,255,0),3);
    #cont[0] = len(approx) 이므로 contour의 포인트 개수를 텍스트로 나타냅니다
    cv2.putText(img_contour, "Points: "+str(con[0]), (x+w//2,y+h//2), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)
    #너비를 텍스트로 나타냅니다
    cv2.putText(img_contour, "Width: "+str(conWH[0]), (x,y), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)
    #높이를 텍스트로 나타냅니다
    cv2.putText(img_contour, "Height: "+str(conWH[1]), (x-w//3,y+h//2), cv2.FONT_HERSHEY_COMPLEX, font_size,font_color,font_thick)