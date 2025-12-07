import os
import cv2
import numpy as np
from Performance_timer import clock

def add_pixel_noise(img, noise_level=0.1, min_val=0.0, max_val=1.0):
    img = img.astype(np.float32) / 255.0  

    rows, cols = img.shape[:2]
    crow, ccol = rows // 2, cols // 2
    r = int(min(rows, cols) * 0.1) 

    mask_lp = np.zeros((rows, cols), np.float32)
    cv2.circle(mask_lp, (ccol, crow), r, 1, -1)
    mask_hp = 1 - mask_lp

    img_lp = np.zeros_like(img)
    img_hp = np.zeros_like(img)

    for c in range(3):
        f = np.fft.fft2(img[:, :, c])
        fshift = np.fft.fftshift(f)

        f_lp = fshift * mask_lp
        f_hp = fshift * mask_hp

        lp = np.fft.ifft2(np.fft.ifftshift(f_lp))
        hp = np.fft.ifft2(np.fft.ifftshift(f_hp))

        lp = np.real(lp)
        hp = np.real(hp)

        img_lp[:, :, c] = lp
        img_hp[:, :, c] = hp

    random_noise_level = np.random.uniform(noise_level, 1-noise_level)
    noisy_img = (img_lp*(1-random_noise_level)) + (img_hp*random_noise_level)
    noisy_img = np.clip(noisy_img, min_val, max_val)
    # 다시 0~255 범위로 변환
    noisy_img = cv2.normalize(noisy_img, None, min_val, max_val, cv2.NORM_MINMAX)
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

if __name__ == "__main__":
    # ---------------- 사용 예시 ----------------
    input_dir = "dataset"          # 원본 폴더
    output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
    apply_noise_to_dataset(input_dir, output_dir, noise_level=0.3) 
