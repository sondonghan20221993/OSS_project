import os
import cv2
import numpy as np

def add_pixel_noise(origin_img, blend_img, noise_level=0., min_val=0.05, max_val=0.95):
    """
    origin_img: 합성될 기본 입력 이미지 (numpy array, 0~255)
    belnd_img: 합성할 이미지 (numpy array, 0~255)
    noise_level: 노이즈 세기 (ex: 0.1일시 belnd이미지의 값 10%만 넣는다.
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    #blend_img 크기가 다를수도 있음으로 resize
    origin_img_shape = (origin_img.shape)
    blend_img = np.resize(blend_img, origin_img_shape)

    # 0~1 정규화
    origin_img = origin_img.astype(np.float32) / 255.0  
    blend_img = blend_img.astype(np.float32) / 255.0 

    # [-noise_level, +noise_level] 범위 랜덤 노이즈
    noisy_img = origin_img + (blend_img*noise_level)

    # [min_val, max_val] 범위로 클리핑
    noisy_img = np.clip(noisy_img, min_val, max_val)

    # 다시 0~255 범위로 변환
    noisy_img = (noisy_img * 255).astype(np.uint8)
    return noisy_img

def apply_noise_to_dataset(input_dir, output_dir, noise_level=0.1):
    os.makedirs(output_dir, exist_ok=True)
    """
    기본이미지에 다른이미지를 불러와 덮어씌워 새로운 이미지를 만드는 방식이다.
    label_list = os.listdir(input_dir)
    blend_label = random.choice([other_label for other_label in label_list if other_label != label])
    위의 코드는 불러온 이미지의 label과 다른 라벨을 불러오는 코드이다.
    ex) rael라벨 이미지 불러옴 -> 위코드 = real라벨이 아닌 라벨

    
    2.
    """
    for label in os.listdir(input_dir):  # cat, dog 같은 라벨 폴더
        label_dir = os.path.join(input_dir, label)
        save_dir = os.path.join(output_dir, label)
        os.makedirs(save_dir, exist_ok=True)
        """
        1. 위 코드로 현재 라벨이 아닌 라벨들을 리스트로 가져온다.
        """
        for fname in os.listdir(label_dir):
            """
            2.blend_path  = 덮어씌울 이미지들의 경로를 input_dir + blend_label을 os함수로 합쳐 만든다.
            3.blend_img_list라는 변수로 2번의 이미지 경로의 모든 이미지 이름을 가져온다.(랜덤으로 가져오기위해)
            4.blend_path os함수 + ramdom.choice로  2 번, 3번 을 합쳐 이미지를 가져온다. 
            5. blend_img = cv2.imread로 4번경로를 읽어온다,
            """

            origin_fpath = os.path.join(label_dir, fname)
            origin_img = cv2.imread(origin_fpath)
            if origin_img is None:
                continue
            noisy = add_pixel_noise(origin_img, blend_img, noise_level=noise_level)
            cv2.imwrite(os.path.join(save_dir, fname), noisy)

    print(f"✅ 모든 이미지에 픽셀 단위 노이즈 적용 완료! (noise_level={noise_level})")

# ---------------- 사용 예시 ----------------
input_dir = "dataset"          # 원본 폴더
output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
apply_noise_to_dataset(input_dir, output_dir, noise_level=0.2)  # 0.2 정도면 꽤 많이 흔들림
