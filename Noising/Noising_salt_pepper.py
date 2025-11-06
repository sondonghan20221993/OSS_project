import os
import cv2
import numpy as np
from Performance_timer import clock

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    """
    img: 입력 이미지 (numpy array, 0~255)
    noise_level: 소금, 후추변환 확률
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  

    # 랜덤픽셀 소금, 후추 노이즈(0 아니면 255)
    noisy_img = img.copy() # 원본 이미지 무수정 복사

    height, width = noisy_img.shape[:2]
    # noise_level 비율만큼 노이즈 개수 계산
    num_noise_pixels = int(height * width * noise_level)
    
    for _ in range(num_noise_pixels):
        #픽셀 좌표 랜덤 선택
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)

        # 검흰 노이즈 적용
        # np.random.choice([0, 1]) 0, 1을 무작위 선택
        noisy_img[y, x] = np.random.choice([0, 1])
    
    # [min_val, max_val] 범위로 클리핑
    noisy_img = np.clip(noisy_img, min_val, max_val)

    # 다시 0~255 범위로 변환
    noisy_img = (noisy_img * 255).astype(np.uint8)
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
#주소에 /가 아닌 \가 들어간다면 주소 앞에 r붙여야함 r"dataset"
#-주의-  경로상에 한국어아 있을시 imread, imwrite가 작동하지 않는다.
input_dir = "dataset"          # 원본 폴더
output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
apply_noise_to_dataset(input_dir, output_dir, noise_level=0.2)  # 0.2 정도면 꽤 많이 흔들림

