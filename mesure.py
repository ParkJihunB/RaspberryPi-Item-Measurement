import cv2
import numpy as np
from laser import Laser 
from Graphic import *
import time

class Measure:
    def __init__(self):
        #카메라의 실제 촬영 크기를 정합니다
        self.frame_w = 800
        self.frame_h = 600
        self.graphic = Graphic(self.frame_w, self.frame_h)
        self.cap = self.graphic.cap #grapic에서 생성한 카메라
        self.laser = Laser()
        self.init_standard() #미리 준비된 기준 및 필터링 사이즈 지정
        self.set_stand_scale() #축척값을 구하는 함수. 이 함수로 축척 값을 저장해도 되고
        #self.scale = 축척값을 입력하세요 #위에서 출력된 축척값을 그대로 입력해도 됩니다
        print("Scale:",self.scale,"  Depth:",self.stand_d)

    def init_standard(self):
        self.stand_w = 5 #기준 물체의 사이즈 지정
        self.stand_h = 5
        self.stand_d = self.laser.measure() #아무 물체도 두지 않았을 때 카메라와 바닥 사이의 거리
        self.min_area = 1000 #측정할 상품의 가장 최소 너비를 지정합니다. 인식 필터링에 사용됩니다
        #이미지를 가공하는 정도가 되는 threshold의 최소와 최대를 정합니다
        #threshold가 너무 크거나 너무 작은 경우 축정 오류가 잦기 때문에 적당히 조절해주세요
        self.max_thr = 200
        self.min_thr = 0
    
    #카메라 화면에 기준 물체 하나를 두어 사이즈를 측정해 scale 값을 가져옵니다
    def set_stand_scale(self):
        self.scale = [1,1]
        #scale이 1일때
        #threshold를 조절하며 기준물체를 측정한 사이즈를 가져옵니다.
        standard_size = self.measure()
        self.scale = self.applay_standard(standard_size[0],standard_size[1])

    #threshold 값에 따라 캡쳐 이미지를 조정하여 상품크기를 구합니다
    #with_standard = True
        #카메라 화면에 기준이 되는 물체를 두어 함께 측정할 경우 사용합니다
        #매 캡쳐마다 기준 물체의 측정 사이즈를 기준으로 축척(scale)을 구합니다
    #with_standard = False
        #미리 축척(scale) 값이 있는 상태에 카메라 화면에 상품 1개만 두어 사용
    def get_result_by_thr_with_standard(self, thr, with_standard = True):
        obj_result = [0,0,0] #상품 정보 저장
        result_count = 0 #20번 반복 측정하여 평균 값을 리턴
        while True:
            success, img = self.cap.read() #영상 캡쳐를 이미지로 읽어옵니다
            img_blur = cv2.GaussianBlur(img, (7,7), 1) #이미지 블러처리
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)#이미지 그레이스케일 처리

            #threshold값을 지정합니다
                #만약 값을 직접 조절할 수 있는 파라미터 창을 이용해 값을 조정하고 싶다면
                #Grapic 클래스의 컨스트럭터에서 init_window 함수를 활성화하고
                #해당 창 이름과 파라미터 이름을 사용해 트랙바 포지션을 값으로 가져올 수 있습니다
                #cv2.getTrackbarPos(트랙바이름, 트랙바가 표시된 창 이름)
                #threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
            threshold1 = thr
            threshold2 = 0
            #이미지를 threshold 값에 따라 조절
            img_canny = cv2.Canny(img_gray, threshold1, threshold2)
            kernel = np.ones((5,5))
            img_dil = cv2.dilate(img_canny, kernel, iterations = 1)
            img_contour = img.copy()

            #이미지에서 상품의 경계 자리를 구한 값을 받아옵니다
            #두개의 폴리곤(기준,상품)을 찾은 경우에만 값을 리턴하도록 합니다(기준/상품)
            #리턴된 값은 너비를 기준으로 큰 순서대로 정렬되어 있습니다
            #찾지 못했거나 폴리곤의 개수가 맞지 않은 경우 0 리턴
            if with_standard: contours = get_contours(img_dil, img_contour,1000, 2)
            #이미 scale 값이 있고 카메라 화면에 상품 1개만 있는 경우
            else: contours = get_contours(img_dil, img_contour,1000, 1)

            #경계값을 찾은 경우
            #   이미지에서 분명한 가장자리를 인식하여 하나의 폴리곤을 구한 경우
            #두개의 상품을 배치한 경우 contours의 크기는 2가 되어야 합니다
            if contours != 0:
                if with_standard: #기준 물체와 함께 촬영하는 경우 기준물체로 scale 값을 구합니다
                    #현재 상품의 크기 >기준의 크기 로 세팅되어 있습니다
                    #리턴된 결과는 너비가 큰 순서대로 이므로: [상품,기준]
                    con = contours[1]
                    #현재 상품 정보를 전송하여 축척을 구합니다
                    self.scale = self.apply_standard(con) 
                    con_wh = self.scale_WH(con,self.scale)
                    #가장자리가 표시된 이미지(img_contour) 위에
                    #con과 con_wh 정보를 텍스트로 그리는 함수
                    draw_con(con, img_contour, con_wh)

                con = contours[0] #상품 정보
                #위에서 구한 축척 정보로 상품 크기를 구합니다
                con_wh = self.scale_WH(con,self.scale)
                draw_con(con, img_contour, con_wh) #상품 정보를 그리기
                depth = self.stand_d - self.laser.measure() #상품 depth 구하기

                #상품 정보의 평균 값을 구합니다.
                if obj_result[0] == 0: #첫번째 루프일 경우
                    obj_result = [con_wh[0],con_wh[1],depth] #그냥 저장
                else: #기존의 평균 값에 현재 값을 더해 다시 평균 구하기
                    obj_result[0] = (obj_result[0] + con_wh[0]) * 0.5
                    obj_result[1] = (obj_result[1] + con_wh[1]) * 0.5
                    obj_result[2] = (obj_result[2] + depth) * 0.5
            
            if result_count > 20: #20번 체크했다면
                return [obj_result[0],obj_result[1],obj_result[2]]
            result_count += 1
            #보고자 하는 각각의 이미지 합쳐서 출력
            #img_stack = stack_images(0.5,([img,img_dil,img_contour]))
            img_stack = stack_images(0.5,([img_contour]))
            
            cv2.imshow("Result", img_stack)
            cv2.resizeWindow('Result', 400,300)

            if cv2.waitKey(1) & 0xff == ord('q'): break

    #상품 측정
    def measure(self):
        result_array = []
        #thr를 조정하며 상품 사이즈를 측정하여 저장합니다
        for i in range(self.min_thr,self.max_thr):
            if i%20 != 0: continue
            result = self.get_result_by_thr_with_standard(i,False)
            area = result[0] * result[1]
            #너비 크기가 0인경우 측정 안 된 것으로 판단
            if area == 0: continue 
            print("At threshold",i,":",result)
            result_array.append(result)
        cv2.destroyAllWindows()
        #저장된 상품 사이즈들을 각각 w,h,d로 나누어 
        #가장 자주 나온 숫자로 골라 리턴
        width_result = []
        height_result = []
        depth_result = []
        for r in result_array:
            #상품 크기를 여러번 측정하여 가장 자주 나온 숫자를 구해야 합니다
            #그러므로 편의상 상품 크기는 반올림하여 int로 처리
            width_result.append(int(np.round(r[0],0)))
            height_result.append(int(np.round(r[1],0)))
            depth_result.append(int(np.round(r[2],0)))
        width = self.get_most_frequent(width_result) #가장 많이 나온 width값 저장
        height = self.get_most_frequent(height_result)#가장 많이 나온 height 저장
        depth = self.get_most_frequent(depth_result)#가장 많이 나온 depth 저장
        print([width,height,depth])
        return[width,height,depth]

    #기준이 되는 물체 정보를 가져와서 축척을 구해 리턴(con 정보 사용)
    def apply_standard(self, stand_con):
        width = stand_con[4][2]
        height = stand_con[4][3]
        scale_w = self.stand_w / width
        scale_h = self.stand_h / height
        return [scale_w, scale_h]
    #기준이 되는 물체 정보를 가져와서 축척을 구해 리턴(wh 정보 사용)
    def applay_standard(self, w, h):
        scale_w = self.stand_w / w
        scale_h = self.stand_h / h
        return [scale_w, scale_h]

    #알아낸 축척 정보로 상품 크기를 구해 리턴
    def scale_WH(self, con, scale):
        width = con[4][2] * scale[0]
        height = con[4][3] * scale[1]
        area = width * height
        con[1] = area
        return [width, height]

    #배열에서 가장 자주 나온 숫자를 구해 리턴
    def get_most_frequent(self, value_array):
        max_value = 0
        max_count = 0
        for v in value_array:
            num = value_array.count(v)
            if num>max_count:
                max_count = num
                max_value = v
        return max_value

M = Measure()
time.sleep(3)
M.measure()