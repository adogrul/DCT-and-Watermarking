import cv2
import numpy as np

baboon_image = cv2.imread('baboon.jpeg', 0)
dct_baboon = cv2.dct(np.float32(baboon_image))

ktu_logo_image = cv2.imread('ktu_logo.jpeg', 0)
dct_ktu_logo = cv2.dct(np.float32(ktu_logo_image))

alpha = 0.1  # Gömme oranı, isteğe bağlı olarak değiştirilebilir
dct_baboon[:dct_ktu_logo.shape[0], :dct_ktu_logo.shape[1]] += alpha * dct_ktu_logo
recovered_ktu_logo = cv2.idct(np.float32(dct_baboon))

cv2.imwrite('watermarked_highfq_baboon.jpeg', np.uint8(recovered_ktu_logo))
cv2.imwrite('dct_ktu_logo.jpeg', np.uint8(dct_ktu_logo))
cv2.imwrite('recovered_ktu_logo_from_highfq.jpeg', ktu_logo_image)

def calculate_psnr(original_path, distorted_path):
    original_image = cv2.imread(original_path)
    distorted_image = cv2.imread(distorted_path)
    assert original_image.shape == distorted_image.shape, "Görüntüler aynı boyutta olmalı"
    mse = np.mean((original_image - distorted_image) ** 2)
    if mse == 0:
        return float('inf')  # Görüntüler aynıysa PSNR sonsuza yakındır
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr
original_path = 'baboon.jpeg'
watermarked_path = 'watermarked_highfq_baboon.jpeg'
psnr_value = calculate_psnr(original_path, watermarked_path)
print("\n\n\nOrijinal ve watermark uygulanmış resim arasındaki PSNR değeri:\n")
print(f"PSNR Değeri: {psnr_value} dB \n\n\n")