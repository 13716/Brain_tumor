import cv2
import numpy as np

# c.greet()
img= cv2.imread(r'C:\Users\hello\Downloads\hand-01.bmp',1)
print(img.shape)
# img = cv2.resize(img, (150, 150))
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # image_smoothed = cv2.GaussianBlur(rgb, (3, 3), 10)
# canny=cv2.Canny(rgb, threshold1=100, threshold2=220)
# kernel= np.ones((1,1),np.uint8)

# new= cv2.erode(canny,kernel,iterations=1)
# # Kiểm tra và giữ lại chỉ các contour là vòng kín
# contours,_ = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # Giữ lại contour lớn nhất
# largest_contour = max(contours, key=cv2.contourArea)
# # Vẽ contour lớn nhất lên hình ảnh
# new = cv2.drawContours(img, [largest_contour], -1, (255, 0, 0), 1)

# # Tạo mặt nạ cho contour lớn nhất
# mask = np.zeros_like(rgb)
# cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# # Gán giá trị 0 cho các vùng không có contour lớn nhất
# img[mask == 0] = 0

cv2.imshow('',img)

# cv2.imshow('',new)
cv2.waitKey(0)
cv2.destroyAllWindows()
