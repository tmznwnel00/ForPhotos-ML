import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 이미지 저장용
import matplotlib.pyplot as plt


def apply_sepia(img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(img, sepia_filter)

def apply_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return img

def apply_vintage(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0.3
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * 0.8
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] * 1.1, 0, 179)
    img_vintage = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    img_vintage = cv2.transform(img_vintage, sepia_filter)
    img_vintage = cv2.convertScaleAbs(img_vintage, alpha=0.9, beta=10)
    noise = np.random.normal(0, 5, img_vintage.shape).astype(np.uint8)
    img_vintage = cv2.add(img_vintage, noise)
    img_vintage = cv2.GaussianBlur(img_vintage, (3, 3), 0.5)
    return img_vintage


# 이미지 로드
try:
    img = cv2.imread('input.jpg')
    if img is None:
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        print("input.jpg 파일을 찾을 수 없어 샘플 이미지를 생성했습니다.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
except:
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    print("input.jpg 파일을 찾을 수 없어 샘플 이미지를 생성했습니다.")

# 필터 적용
img_sepia = apply_sepia(img)
img_grayscale = apply_grayscale(img)
img_vintage = apply_vintage(img)

# 결과 시각화 및 저장
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(img)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_sepia)
axes[0, 1].set_title('sepia')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_grayscale, cmap='gray')
axes[1, 0].set_title('grayscale')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_vintage)
axes[1, 1].set_title('vintage')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('filters_result.png', dpi=300, bbox_inches='tight')
print("시각화 결과가 'filters_result.png'로 저장되었습니다.")
