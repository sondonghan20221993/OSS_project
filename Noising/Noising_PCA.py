from sklearn.decomposition import PCA
import os
import cv2
import numpy as np  

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    # 0~1 범위로 정규화
    img_norm = img.astype(np.float32) / 255.0

    # PCA 객체 생성. n_components가 0과 1 사이의 float이면,
    # 분산의 `noise_level` 비율만큼을 유지하는 데 필요한 주성분 개수를 자동으로 선택합니다.
    pca = PCA(n_components=noise_level)

    # 이미지가 컬러인지 흑백인지 확인
    if len(img.shape) == 3: # 컬러 이미지 (높이, 너비, 채널)
        # B, G, R 채널로 분리
        b, g, r = cv2.split(img_norm)
        
        # 각 채널에 대해 PCA 적용 및 복원
        b_pca = pca.inverse_transform(pca.fit_transform(b))
        g_pca = pca.inverse_transform(pca.fit_transform(g))
        r_pca = pca.inverse_transform(pca.fit_transform(r))
        
        # 처리된 채널들을 하나로 합침
        noisy_img = cv2.merge([b_pca, g_pca, r_pca])

    else: # 흑백 이미지 (높이, 너비)
        # 이미지 자체에 PCA 적용 및 복원
        noisy_img = pca.inverse_transform(pca.fit_transform(img_norm))

    # [min_val, max_val] 범위로 클리핑
    noisy_img = np.clip(noisy_img, min_val, max_val)

    # 다시 0~255 범위로 변환
    noisy_img = (noisy_img * 255).astype(np.uint8)
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
apply_noise_to_dataset(input_dir, output_dir, noise_level=0.96)  # 0.96 정도면 꽤 많이 흔들림

