import cv2
import os
import time
import functools
from multiprocessing import Pool, cpu_count
import numpy as np
import importlib
#--------------------------------------------- 데커레이터-----------------------------------------------#
"""
시간효율 확인하기 위해 수업에서 배운 데커레이터를 이용해서 시간을 측정해보자
"""

#--------------------------------------------이미지 경로들 읽어오는 함수---------------------------------#
def get_img_path(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_path = []#작업할 이미지 경로 저장
    print("작업 목록 생성 중...")
    for label in os.listdir(input_dir):  # cat, dog 같은 라벨 폴더
        label_dir = os.path.join(input_dir, label)
        if not os.path.isdir(label_dir): # 폴더가 아니면 건너뛰기
            continue
            
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True) #저장 경로 폴더생성
        count = 0
        for fname in os.listdir(label_dir):
            all_path = os.path.join(label_dir, fname) #전체경로 추출
            img_path.append(all_path)
            count +=1
            if(count == 100): break
    return img_path
#--------------------------------------------함수 가져오기----------------------------------------------#
def get_library(a):
    module = importlib.import_module(a)
  
#---------------------------가져온 함수 + 이미지 경로 리스트로 이미지생성, 저장--------------------------#
def load_noise_maker(img_path, fun, output_dir):
    #cv2로 이미지 읽기
    img = cv2.imread(img_path)
    #noise함수로 noise_img반환
    noisy_img = fun(img)
    #label명 추출 + 저장경로 생성
    parent_folder = os.path.basename(os.path.dirname(img_path))
    save_dir = os.path.join(output_dir, parent_folder)
    os.makedirs(save_dir, exist_ok=True)
    #이름 
    save_name = os.path.basename(img_path)
    save_path = os.path.join(save_dir, save_name)    
    cv2.imwrite(save_path, noisy_img)
    return

@clock
def main():
    #경로 지정
    input_dir = "D:/archive/deepfake_database/train" 
    output_dir = r"D:\check_salt_pepper_test" 
    work_img_path = get_img_path(input_dir, output_dir)
    #작업 노이즈 함수 지정
    use_noise_fun = get_library("Noising") #-> 사용할 노이즈함수 파일이름
    
    #멀티프로세싱 과정
    num_cores = cpu_count() 
    print(num_cores) 
    start_time = time.time() 
    work_fun = functools.partial(use_noise_fun.add_pixel_noise, #-> 코드중 변형이미지 return하는 함수만 사용
                                    noise_level=0.1,
                                      min_val=0.05,
                                      max_val=0.95) 
    work_fun = functools.partial(load_noise_maker,  
                                fun = work_fun, 
                                output_dir = output_dir)
    with Pool(processes=num_cores) as pool: 
        pool.map(work_fun, work_img_path)  
        print("t")
if __name__ == "__main__":
    main()
"""
사용시 cmd에서 사용해야 실행가능
"""