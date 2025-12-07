import os
import cv2
import numpy as np
import math
from Performance_timer import clock

def _bilinear_interpolate(img, x, y):
    """
    내부 보간 함수 (빈 공간 채우기용)
    img: 입력 이미지 (numpy array, 0~255)
    x, y: 부동 소수점 좌표
    """
    h, w, c = img.shape

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def add_pixel_noise(img, noise_level=0.1):
    """
    img: 입력 이미지 (numpy array, 0~255)
    noise_level: 0.0~1.0 사이의 회전 각도 비율 (예: 0.1 → 10% 회전)
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  
    h, w, c = img.shape

    # [-noise_level, +noise_level] 범위 랜덤 회전 비율 생성
    MAX_ROT_ANGLE = 360.0  # 최대 회전 각도 (도 단위)
    angle = noise_level * MAX_ROT_ANGLE
    rad = math.radians(angle)

    cos_t = math.cos(rad)
    sin_t = math.sin(rad)

    cx = w / 2.0
    cy = h / 2.0

    # 회전 결과 저장할 빈 이미지 생성
    rotated_img = np.zeros_like(img, dtype=np.float32)

    # 좌표 회전 + 보간
    for y_prime in range(h):
        for x_prime in range(w):
            tx = x_prime - cx
            ty = y_prime - cy

            x =  cos_t * tx + sin_t * ty + cx
            y = -sin_t * tx + cos_t * ty + cy

            if 0 <= x < w - 1 and 0 <= y < h - 1:
                rotated_img[y_prime, x_prime] = _bilinear_interpolate(img, x, y)

    # 다시 0~255 범위로 변환
    rotated = (rotated_img * 255).astype(np.uint8)

    return rotated

@clock
def apply_noise_to_dataset(input_dir, output_dir, noise_level=0.1):
    os.makedirs(output_dir, exist_ok=True)

    for label in os.listdir(input_dir):  # cat, dog 같은 라벨 폴더
        label_dir = os.path.join(input_dir, label)
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)

        for fname in os.listdir(label_dir):
            origin_fpath = os.path.join(label_dir, fname)
            origin_img = cv2.imread(origin_fpath)
            if origin_img is None:
                continue
            noisy = add_pixel_noise(origin_img, noise_level=noise_level)
            cv2.imwrite(os.path.join(save_dir, fname), noisy)

    print(f"✅ 모든 이미지에 픽셀 단위 노이즈 적용 완료! (noise_level={noise_level})")
if __name__ == "__main__":
    # ---------------- 사용 예시 ----------------
    input_dir = "dataset"          # 원본 폴더
    output_dir = "dataset_rotated"   # 노이즈 추가된 폴더
    apply_noise_to_dataset(input_dir, output_dir, noise_level=0.3)
