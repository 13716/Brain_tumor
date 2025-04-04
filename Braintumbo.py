import cv2
import numpy as np
from tkinter import Tk, Button, Label, Entry
from tkinter.filedialog import askopenfilename
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter 

def process_image():
    path = askopenfilename(title="Chọn hình ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")])
    if not path:
        return

    img = cv2.imread(path)
    cv2.imshow('image_original',img)
    img = cv2.resize(img, (338, 287))
    
    # Chuyển đổi sang màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Cải thiện chất lượng hình ảnh
    # 1. Điều chỉnh độ tương phản
    alpha = 0.5# Hệ số điều chỉnh độ tương phản
    beta = 0    # Giá trị điều chỉnh độ sáng
    contrast_adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 2. Cân bằng histogram
    equalized_image = cv2.equalizeHist(contrast_adjusted)
    cv2.imshow('CBTP',equalized_image)
    # 3. Làm mịn hình ảnh
    image_smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if gray is None:
        raise ValueError("Image not loaded. Check the file path.")
    data = np.array(image_smoothed).reshape((-1, 1))
    # data = np.array(equalized_image).reshape((-1, 1))
    k = int(k_entry.get())
    kmeans = KMeans(n_clusters=k, init='random', n_init=1) #max_iter=100, random_state=42
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.astype(np.uint8)
    
    segmented_image = centroids[labels].reshape(image_smoothed.shape)  # Khôi phục lại bức ảnh bằng cách ánh xạ nhãn về màu
    segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)  # Chuyển đổi kiểu dữ liệu
    # if np.mean(centroids[0]) > np.mean(centroids[1]):
    #     binary_img = np.where(labels.reshape(image_smoothed.shape[:2]) == 0, 255, 0).astype(np.uint8)
    # else:
    #     binary_img = np.where(labels.reshape(image_smoothed.shape[:2]) == 1, 255, 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    # new = cv2.erode(segmented_image, kernel, iterations=1)
    new = cv2.erode(segmented_image, kernel, iterations=1)
    img2 = np.zeros_like(img).astype(np.uint8)
    print(path[25:])
    print(centroids)
    for i in range(k):
        clusters_points = data[labels == i].flatten()
        print(f"\n Nhóm {i + 1} {clusters_points}")

        # Kiểm tra điều kiện tâm giá trị và giá trị trong nhóm
        if ((centroids[i] > 95) & (centroids[i] <= 200)):
            if np.all(clusters_points > 90) & np.all(clusters_points < 139):
                print("a")
                img2[((new >= 90) & (new < 130))] = 255
            elif np.all(clusters_points >= 130) & np.all(clusters_points <= 255):
                print("b")
                img2[((new >= 135) & (new <= 255))] = 255
        elif centroids[i] > 200:
            if np.any(clusters_points > 150):
                print("c")
                img2[((new > 150) & (new <= 255))] = 255
    # img2[new == 255] = 255
    # img2[((new >= 135) & (new <= 255))] = 255 
    img2_gray = img2[:, :, 0]  # Lấy kênh đơn sắc từ ảnh (giả sử img2 là ảnh nhị phân)
    # cv2.imshow('', img2_gray)

    # Tìm các vùng liên thông
    num_labels, labels_img = cv2.connectedComponents(img2_gray)
    print(num_labels)
    # Thêm điều kiện để phân đoạn khối u
    if num_labels >= 1:  # Kiểm tra nếu có ít nhất một vùng
        # Tính toán kích thước của mỗi vùng
        sizes = np.bincount(labels_img.flatten())
        sizes[0] = 0  # Bỏ qua nhãn 0 (nền)
        print(sizes)
        # Xác định vùng có kích thước lớn nhất
        largest_label = np.argmax(sizes)

        # Tạo mask cho vùng lớn nhất
        largest_region_mask = (labels_img == largest_label) & (img2_gray == 255)

        # Đổi màu các điểm ảnh trong ảnh gốc sang đỏ nếu chúng thuộc vùng lớn nhất trong img2
        img[largest_region_mask] = [0, 0, 255]

    cv2.imshow('Image_after', img)
    cv2.imshow('smoothed', image_smoothed) 
    # cv2.imshow('binary Image', binary_img) 
    cv2.imshow('Segmented Image', segmented_image)
    cv2.imshow('image erode', new)
    cv2.imshow('new', img2)

    # Hiển thị bức ảnh đã phân nhóm
    cv2.waitKey(0)  # Đợi phím nhấn
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ

# Tạo giao diện
root = Tk()
root.title("Image Segmentation Tool")

label = Label(root, text="Chọn hình ảnh để phân nhóm:")
label.pack(pady=10)

k_label = Label(root, text="Nhập giá trị k:")
k_label.pack(pady=5)

k_entry = Entry(root)  # Thêm trường nhập liệu cho k
k_entry.pack(pady=5)

button = Button(root, text="Chọn Hình Ảnh", command=process_image)
button.pack(pady=20)

root.mainloop()
