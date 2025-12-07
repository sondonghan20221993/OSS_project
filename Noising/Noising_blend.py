import os
import cv2
import numpy as np
import random
from Performance_timer import clock

def add_pixel_noise(origin_img, blend_img, noise_level=0., min_val=0.05, max_val=0.95):
    """
    origin_img: 합성될 기본 입력 이미지 (numpy array, 0~255)
    belnd_img: 합성할 이미지 (numpy array, 0~255)
    noise_level: 노이즈 세기 (ex: 0.1일시 belnd이미지의 값 10%만 넣는다.
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    h, w = origin_img.shape[:2]
    blend_img = cv2.resize(blend_img, (w, h))
    # 0~1 정규화
    origin_img = origin_img.astype(np.float32) / 255.0  
    blend_img = blend_img.astype(np.float32) / 255.0 

    # [-noise_level, +noise_level] 범위 랜덤 노이즈
    noisy_img = origin_img*(1-noise_level) + (blend_img*noise_level)

    # [min_val, max_val] 범위로 클리핑
    noisy_img = np.clip(noisy_img, min_val, max_val)

    # 다시 0~255 범위로 변환
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return noisy_img

@clock
def apply_noise_to_dataset(input_dir, output_dir, noise_level=0.1):
    os.makedirs(output_dir, exist_ok=True)
    label_list = os.listdir(input_dir)
    print(label_list)
    for label in os.listdir(input_dir):  # cat, dog 같은 라벨 폴더
        label_dir = os.path.join(input_dir, label)
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        blend_label = random.choice([other_label for other_label in label_list if other_label != label])#ChatGPT 사용

        for fname in os.listdir(label_dir):
            blend_fpath = os.path.join(input_dir, blend_label)
            blend_img_list = os.listdir(blend_fpath)
            blend_fpath = os.path.join(blend_fpath, random.choice(blend_img_list))
            print(blend_fpath)
            blend_img = cv2.imread(blend_fpath)
            origin_fpath = os.path.join(label_dir, fname)
            print(origin_fpath)
            origin_img = cv2.imread(origin_fpath)
            if origin_img is None:
                continue
            noisy = add_pixel_noise(origin_img, blend_img, noise_level=noise_level)
            cv2.imwrite(os.path.join(save_dir, fname), noisy)
    print(f"✅ 모든 이미지에 픽셀 단위 노이즈 적용 완료! (noise_level={noise_level})")

if __name__ == "__main__":
    # ---------------- 사용 예시 ----------------
    input_dir = "dataset"          # 원본 폴더
    output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
    apply_noise_to_dataset(input_dir, output_dir, noise_level=0.2)  # 0.2 정도면 꽤 많이 흔들림
