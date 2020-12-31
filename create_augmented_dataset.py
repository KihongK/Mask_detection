import os
from os import listdir
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageFile

__version__ = '0.3.0'


IMAGE_DIR = os.getcwd()+"\\images" # images 에는 마스크 사진이 3장 저장되어있다.
MASK_IMAGES=["default-mask.png","black-mask.png","blue-mask.png"]

MASK_IMAGE_PATHS =[os.path.join(IMAGE_DIR, mask_image) for mask_image in MASK_IMAGES] # 3장의 마스크에 대한 위치

FACE_FOLDER_PATH=os.getcwd()+"\\dataset\\without_mask" # 마스크 없는 인물 사진
AUGMENTED_MASK_PATH=os.getcwd()+"\\dataset\\with_mask" # 마스크 있는 인물 사진



def create_mask(face_path, mask_path, augmented_mask_path, color):
    show = False
    model = "hog"
    FaceMasker(face_path, mask_path, augmented_mask_path, color=color).mask()



class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
        
    
    def __init__(self, face_path, mask_path, augmented_mask_path, show=False, model='hog', color="default"):
        # model = 'hog' 는 CPU를 사용할때 사용하는 모델로 낮은 정확도를 갖지만 속도가 빠릅니다. (default)
        #         'cnn' 은 GPU가 사용가능할때 적용가능합니다. 높은 정확도를 갖습니다.
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self.augmented_mask_path=augmented_mask_path
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.color=color
        
    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path) # 이미지파일 load (array 형식으로 저장)
        
        face_locations = face_recognition.face_locations(face_image_np, model=self.model) 
        # load한 이미지의 얼굴에 해당하는 bounding box를 찾습니다. (x1, y1, x2, y2)
        
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        # 얼굴의 'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge' ... 등등 의 좌표를 저장합니다. (각 부위 별 좌표의 개수는 다릅니다!)
        
        self._face_img = Image.fromarray(face_image_np) # 호출한 이미지(array) 를 이미지 형식으로 변환
        self._mask_img = Image.open(self.mask_path).convert("RGBA") # 이미지 형식으로 호출

        found_face = False
        # 사진속 얼굴에서 턱부분과 코부분이 없어도 얼굴로 인식하도록
        for face_landmark in face_landmarks: # ** face_landmarks 의 구조를 확인해보세요!!
            # check whether facial features meet requirement
            skip = False
            
            # KEY_FACIAL_FEATURES = ('nose_bridge', 'chin') 코 브릿지와 턱부분을 감지 못한다 == 마스크를 쓰고있다.
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True # 코와 턱이 없다면 skip을 true로 하고 반복문 종료
                    break
           
            # 이 아래는 코와 턱이 있는 이미지를 대상으로 하는 과정입니다.
            if skip:
                continue
            found_face = True
            self._mask_face(face_landmark) # 

        if found_face:
            if self.show:
                self._face_img.show()

            # save
            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge'] # 리스트(value 4개 고정)
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4] # 4개 좌표중 2번쨰것을 선택한다는 의미
        nose_v = np.array(nose_point)

        chin = face_landmark['chin'] # 리스트(value 17개)
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2] # 중앙 좌표 선택
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8] # 가장 왼쪽에 있는 좌표
        chin_right_point = chin[chin_len * 7 // 8] # 가장 오른쪽에 있는 좌표 선택

        # split mask and resize 마스크 이미지를 load한 사람 이미지의 코와 입을 가릴 수 있도록 사이즈 변경
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        new_face_path=self.augmented_mask_path+"\\with-mask-"+self.color+"-"+self.face_path.split("\\")[-1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)

'''
print(IMAGE_DIR)  # 이미지 저장 위치
print(MASK_IMAGES) # 마스크 이미지 파일 명
print(MASK_IMAGE_PATHS) # 마스크 이미지 저장 위치와 파일명

# 인물 사진(경로) 
# *목표* : 마스크없는 인물 사진에 face_recognition 을 통해 마스크 씌우는 효과를 줌

print(FACE_FOLDER_PATH) # without 마스크
print(listdir(FACE_FOLDER_PATH))
print(AUGMENTED_MASK_PATH) # with 마스크(빈 폴더 without 마스크 인물 이미지에  마스크를 씌운 이미지로 변환하고 저장할 곳)
print(COLOR) # 마스크 정보(색상)
'''

if __name__ == '__main__':
    for MASK_IMAGE_PATH in MASK_IMAGE_PATHS: # 마스크 종류가 반복됩니다
        # 이미지에 3개 마스크를 적용할 예정 (순서: 일반 마스크 전체 이미지에 적용 -> 검정색 마스크 전체이미지에 적용 -> 마지막 마스크 전체 이미지에 적용)
        COLOR=MASK_IMAGE_PATH.split("\\")[-1].split(".")[0]
        FACE_IMAGE_PATHS=[os.getcwd()+"\\dataset\\without_mask\\"+path for path in listdir(FACE_FOLDER_PATH)]
        
        for FACE_IMAGE_PATH in FACE_IMAGE_PATHS:
            print("face image path: ",FACE_IMAGE_PATH)
            create_mask(FACE_IMAGE_PATH, MASK_IMAGE_PATH, AUGMENTED_MASK_PATH, COLOR)
            # FACE_IMAGE_PATH: 변환할 이미지(경로 + 파일명)
            # MASK_IMAGE_PATH: 적용할 마스크 이미지 (경로 + 파일명) ** 위 for 문에서 MASK_IMAGE_PATHS 를 통해 나온 겁니다.
            # AUGMENTED_MASK_PATH: 마스크가 적용된 이미지 저장 위치
            # COLOR: MASK_IMAGE 의 파일명(적용할 마스크의 파일명이 색상을 의미합니다)
