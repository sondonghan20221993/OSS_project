import os
import cv2
import numpy as np

def add_pixel_noise(img, noise_level=0.1, min_val=0.05, max_val=0.95):
    """
    img: 입력 이미지 (numpy array, 0~255)
    noise_level: 소금, 후추변환 확률
    min_val, max_val: 픽셀 값 클리핑 범위 (정규화된 상태에서)
    """
    # 0~1 정규화
    img = img.astype(np.float32) / 255.0  

    # 랜덤픽셀 소금, 후추 노이즈(0 아니면 255)
    noisy_img = img 
    """
    해야할것 noise_level만큼  소금, 후추 잡음을 적용 시키는것이다. ex-> noise_level = 0.1이면 10%센트로 적용을 시키는것 -> 20 * 20이미지라면 총 픽셀수는 1000개 그러므로 100개만 적용시키면 된다.(R, G, B) 3개의 채널이 400개씩 가지고 있다.
    choice 함수로 [0, 1] #아얘 흑백이거나, 아얘 밝게 바꾸는것이다.

    그래서 위 예시로 40픽셀을 바꿔야하면 
    for i in range(40):으로 40번 반복 
    img(높이, 너비, 채널)이다.
        컬러 이미지이므로 채널의 범위는 0,1,2(R,G,B) 3개이다.
        
        결론: height, width, chanel 
        img.shape로 모두 알수있다.
        
        그걸로 각 범위를 지정해 choice로 골라주자. ex) -> (20,30,3) =  높이 20픽셀 너비 30픽셀 채널 3   choice([0:19])로 높이, choice([0:29])로 너비, choice([0:2]) 채널을 랜덤 변수로 뽑아
        img[랜덤높이, 랜덤너비, 랜덤채널] = choice([0,255])로 하면

    120개의 무작위 픽셀을 검은색 OR 흰색으로 바꾸게 된다. 이 바꾼값을  return 시켜보자

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
#주소에 /가 아닌 \가 들어간다면 주소 앞에 r붙여야함 r"dataset"
#-주의-  경로상에 한국어아 있을시 imread, imwrite가 작동하지 않는다.
input_dir = "dataset"          # 원본 폴더
output_dir = "dataset_noisy"   # 노이즈 추가된 폴더
apply_noise_to_dataset(input_dir, output_dir, noise_level=0.2)  # 0.2 정도면 꽤 많이 흔들림
