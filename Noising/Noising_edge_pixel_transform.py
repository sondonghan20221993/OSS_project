import os
import cv2
import numpy as np
from Performance_timer import clock

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    """
    img: 입력 이미지 (numpy array, 0~255)
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  
    
    ## 이미지 정보 추출(y,x,color)
    img_color = img.shape[2]
    img_col = img.shape[1]
    img_row = img.shape[0]
    
    #탐색 범위(얼굴줌심으로 하기위해 범위 조정)
    start_row = int(img_row * 0.3)
    end_row = int(img_row * 0.7)
    start_col = int(img_col * 0.3)
    end_col = int(img_col * 0.7)

    resize_img = img[start_row:end_row, start_col:end_col, :]

    # numpy로 연산을 위한 numpy배열 선언
    sum_neighbors = np.zeros_like(resize_img)
    count_neighbors = np.zeros_like(resize_img)
    check_neighbors = np.zeros_like(resize_img)

    # 아래 연산
    copy_img = resize_img.copy()
    sum_neighbors[:-1,:,:] += resize_img[1:,:,:]  
    count_neighbors[:-1,:,:] += 1
    copy_img[:-1,:,:] -= resize_img[1:,:,:]
    check_neighbors[:-1,:,:][abs(copy_img[:-1,:,:]) >= noise_level] = 1

    # 위 연산
    copy_img = resize_img.copy()
    sum_neighbors[1:,:,:] += resize_img[:-1,:,:]
    count_neighbors[1:,:,:] += 1
    copy_img[1:,:,:] -= resize_img[:-1,:,:]
    check_neighbors[1:,:,:][abs(copy_img[1:,:,:]) >= noise_level] = 1

    # 왼쪽 연산
    copy_img = resize_img.copy()
    sum_neighbors[:,1:,:] += resize_img[:,:-1,:]
    count_neighbors[:,1:,:] += 1
    copy_img[:,1:,:] -= resize_img[:,:-1,:]
    check_neighbors[:,1:,:][abs(copy_img[:,1:,:]) >= noise_level] = 1

    # 오른쪽 연산
    copy_img = resize_img.copy()
    sum_neighbors[:,:-1,:] += resize_img[:,1:,:]
    count_neighbors[:,:-1,:] += 1
    copy_img[:,:-1,:] -= resize_img[:,1:,:]
    check_neighbors[:,:-1,:][abs(copy_img[:,:-1,:]) >= noise_level] = 1

    #-------------------------------------변형과정-----------------------------------------#
    noised = img.copy() # 원본이미지 복사
    mask = (check_neighbors==1) & (count_neighbors>0)#resize_img에서 바꿀 픽셀값 정보
    region = noised[start_row:end_row, start_col:end_col].copy() # 노이즈 적용할 영역 복사
    h, w, c = region.shape # 영역 크기 정보
    pixel_size = 12  # ← 픽셀 크기 (값 키울수록 네모 큼)
    # 블록화
    small = cv2.resize(region, (w//pixel_size, h//pixel_size), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    region[mask] = pixelated[mask] # 노이즈 감지된 부분만 블록 적용
    noised[start_row:end_row, start_col:end_col] = region # 노이즈 적용된 영역 복사
    noisy_img = (noised * 255).astype(np.uint8)
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

if __name__ == "__main__":
    # ---------------- 사용 예시 ----------------
    input_dir = "dataset"          # 원본 폴더
    output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
    apply_noise_to_dataset(input_dir, output_dir, noise_level=0.03)  # 0.03 정도면 일부 픽셀 변형됨

