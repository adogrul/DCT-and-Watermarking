import cv2
import numpy as np

baboon_image = cv2.imread('baboon.jpeg', 0)
dct_baboon = cv2.dct(np.float32(baboon_image))

ktu_logo_image = cv2.imread('ktu_logo.jpeg', 0)
dct_ktu_logo = cv2.dct(np.float32(ktu_logo_image))

# Orta frekans bandını seçin
mid_band_size = (dct_baboon.shape[0] // 2, dct_baboon.shape[1] // 2)
alpha = 0.1  # Gömme oranı, isteğe bağlı olarak değiştirilebilir
# Logoyu orta frekans bandına gömün
dct_baboon[:mid_band_size[0], :mid_band_size[1]] += alpha * dct_ktu_logo

# İşaretlenmiş resmi kaydedin
watermarked_image = cv2.idct(np.float32(dct_baboon))
cv2.imwrite('watermarked_midfq_baboon.jpeg', np.uint8(watermarked_image))


# PSNR değerini yazdırın
#print("\n\n\nOrijinal ve watermark uygulanmış resim arasındaki PSNR değeri:\n")
#print(f"PSNR Değeri: {psnr_value} dB \n\n\n")