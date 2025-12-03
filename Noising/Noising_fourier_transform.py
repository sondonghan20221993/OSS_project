import os
import cv2
import numpy as np
from Performance_timer import clock

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    #이미지 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  
    noise_level *= 10
    """
    1.주파수 마스크 생성(저주파, 고주파)

    2.변환 과정
        fft2 푸리에 변환

        fftshift → 저주파를 중앙으로 이동

        mask_lp, mask_hp 별로 나누기

        역변환 ifft2 해서 복원

        실수부 활용(np.real) LPF / HPF별 생성

    3.원본 복원

    4. 이미지 시각화
    """

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
input_dir = "dataset"          # 원본 폴더
output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
apply_noise_to_dataset(input_dir, output_dir, noise_level=0.2)  # 0.2 정도면 꽤 많이 흔들림
