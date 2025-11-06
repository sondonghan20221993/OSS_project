import os
import cv2
import numpy as np
from queue import Queue

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    """
    img: 입력 이미지 (numpy array, 0~255)
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  
    """
    기존_아이디어
    탐색과정-
    모든 픽셀을 방문 -> 해당 픽셀에서 위,아래, 왼쪽, 오른쪽 확인-> 차이가 많이 난다면 해당 위치 기록

    변형과정-
    기록된 위치 하나하나 접근 -> (현재픽셀값 + (위,아래,왼,오픽셀값)/4)/2로 픽셀값을 변형한다.

    문제점: for문으로 구현시 3중반복문, 상당히 비효율적이다. 그래서 numpy연산으로 바꾸어 
    비교적 빠르게  탐색,변형을 하기로 했다.

    개선_아이디어
    탐색과정-
    이미지 배열만큼 배열(numpy)생성 -> 이미지 shift후 대입(#아래 연산 예로 들어본다.)

    이미지 배열(행기준)   -   바꿀값(행기준)
    1         ->      2
    2         ->      3
    3         ->      4
    4         ->      5

    이렇게 원하는 방향(아래로)으로 shift시킨 이미지를 매핑시켜주면
    ex) 1번픽셀에 2번값이 들어가게됨  
    -> 그러면 다시 이미지를 복사해 1번 픽셀 - shift이미지 하면 결국 아래픽셀과의 차를 구할 수 있다.

    #아래 연산 세부 설명
    1.연산을 위해 임시로 이미지 복사
    2.위 처럼 shift시켜 픽셀값을 구함(sum_neighbors에 resize_img를 시프트 시켜 대입)
    3.픽셀값 나누기 위해 갯수도 더함(count_neighbors에 +1해서 기록)
    4.1번 이미지 - 2번 이미지(복사이미지 - shift이미지)로 픽셀차를 구함
    5.픽셀차가 일정이상일시 기록(여기서는 0.1이상 날시)


    최종적으로 아래, 위, 왼쪽, 오른쪽 연산이 끝나면 변형하게 된다.

    세부 설명[:,:,:]에서 #[y,x,color]을 나타낸다.
    [:-1,:,:] = [1:,:,:]  -> y(1,2,3) = y(2,3,4)로 매핑해 시프트 시킨다는 아이디어를 이해하면 된다.

    +매 연산시 
    copy_img는 새로 선언
    count_neighbors, check_neighbors, sum_neighbors 누적해서 처리하는것이라 새로선언X
    
    #나름 열심히 설명했지만 이해 안된다면 여쭤보셔도 됩니다.
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  
    
    ## 이미지 정보 추출(y,x,color)
    img_color = img.shape[2]
    img_col = img.shape[1]
    img_row = img.shape[0]
    
    #탐색 범위(얼굴줌심으로 하기위해 범위 조정)
    start_row = int(img_row*0.3)
    end_row = int(img_row*0.7)
    start_col = int(img_col*0.3)
    end_col = int(img_col*0.7)

    resize_img = img[start_row:end_row, start_col:end_col, :]

    #numpy로 연산을 위한 numpy배열 선언
    sum_neighbors = np.zeros_like(img[start_row:end_row, start_col:end_col, :])
    count_neighbors = np.zeros_like(img[start_row:end_row, start_col:end_col, :])
    check_neighbors = np.zeros_like(img[start_row:end_row, start_col:end_col, :])

    #아래 연산 
    copy_img = resize_img.copy()
    sum_neighbors[:-1,:,:] += resize_img[1:,:,:]  
    count_neighbors[:-1,:,:] += 1
    copy_img[:-1,:,:] -= resize_img[1:,:,:] 
    check_neighbors[:-1,:,:][abs(copy_img[:-1,:,:]) >=0.1] = 1
    """
    위쪽, 왼쪽, 오른쪽 연산 구현하기
    """

    #-------------------------------------변형과정-----------------------------------------#
    noised = img.copy()#원본이미지 복사
    mask = (check_neighbors==1) & (count_neighbors>0)#resize_img에서 바꿀 픽셀값 정보
    origin_mask = np.zeros_like(noised, dtype=bool)# 원본이미지와 resize_img는 사이즈가 다르기에 크기 맞추기위한 배열선언
    origin_mask[start_row:end_row, start_col:end_col,:] = mask #기존 마스크 매핑
    noised[origin_mask] = (noised[origin_mask] + (sum_neighbors[mask]  / count_neighbors[mask]))/2 #이미지 변형
    noisy_img = noised
    # 다시 0~255 범위로 변환
    noisy_img =(noisy_img * 255).astype(np.uint8)
    return noisy_img

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
