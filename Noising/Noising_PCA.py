from sklearn.decomposition import PCA
import os
import cv2
import numpy as np  

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    """
    img: 입력 이미지 (numpy array, 0~255)
    noise_level: 노이즈 세기
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    """
    주성분 분석:
    이미지의 각 색상 채널(R, G, B)에 대해 PCA(주성분 분석)을 적용해서 해상도를 줄이고 다시 복원하는 과정

    코드자체는 짧아서 과정만 설명
    PCA(n_components=0.97) -> 데이터 분산이 97%를 유지할 때 필요한 주성분 개수(데이터 97%만 남김)
    pca.fit_transform -> 차원을 축소한다
    pca.inverse_transform -> 축소한 차원을 다시 복원시킨다(이미지 흐려짐)

    #주의점# 
    color채널에 대한 PCA가 아닌 이미지 픽셀에 대한 PCA로 해야합니다.
    """

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
