import os
import cv2
import numpy as np
from queue import Queue
from Performance_timer import clock

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    """
    img: 입력 이미지 (numpy array, 0~255)
    noise_level: 노이즈 세기 (0~1, 예: 0.1 → 최대 ±0.1 변화)
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  
    
    # 이미지 size 저장
    img_color = img.shape[2]
    img_col = img.shape[1]
    img_row = img.shape[0]
    #탐색범위 지정
    find_direction = ((1,0), (-1,0), (0,-1), (0,1))
    coordinate_q = Queue()
    #탐색 범위(얼굴줌심)
    start_row = int(img_row*0.1)
    end_row = int(img_row*0.9)
    start_col = int(img_col*0.2)
    end_col = int(img_col*0.8)
    
    for color in range(img_color):
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                #현재 픽셀
                now_pixel = img[row, col, color]
                for row_r, col_c in find_direction:
                    if(row + row_r>= img_row or row + row_r<0 or col+col_c >= img_col or col+col_c<0):
                        continue
                    else:
                        compare_pixel = img[row + row_r, col+col_c, color]
                        if(abs(now_pixel-compare_pixel)>=noise_level):
                            coordinate_q.put([row, col, color])
                            break
    #변형
    while(coordinate_q.empty() == False):
        row, col, color = coordinate_q.get()
        sum_pixel =0
        count = 0
        for row_r, col_c in find_direction:
            if(row + row_r>= img_row or row + row_r<0 or col+col_c >= img_col or col+col_c<0):
                continue
            else:
                sum_pixel += img[row + row_r, col+col_c, color]
                count +=1
        sum_pixel /= count
        if(img[row, col, color] > sum_pixel):
            img[row, col, color] +=sum_pixel* 0.1
        else:
            img[row, col, color] +=sum_pixel* 0.1
    noisy_img = img
    # 다시 0~255 범위로 변환
    noisy_img =(noisy_img * 255).astype(np.uint8)
    return noisy_img

@clock
def apply_noise_to_dataset(input_dir, output_dir, noise_level=0.1):
    os.makedirs(output_dir, exist_ok=True)

    for label in os.listdir(input_dir):  # cat, dog 같은 라벨 폴더
        label_dir = os.path.join(input_dir, label)
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            noisy = add_pixel_noise(img, noise_level=noise_level)
            cv2.imwrite(os.path.join(save_dir, fname), noisy)

    print(f"✅ 모든 이미지에 픽셀 단위 노이즈 적용 완료! (noise_level={noise_level})")

# ---------------- 사용 예시 ----------------
input_dir = "dataset"          # 원본 폴더
output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
apply_noise_to_dataset(input_dir, output_dir, noise_level=0.03)  # 0.03 정도면 일부 픽셀 변형됨
