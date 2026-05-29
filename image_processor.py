import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    Kelas untuk menangani seluruh logika operasi pengolahan citra.
    Setiap metode statis atau metode instans yang menerima gambar akan mengembalikan gambar baru (hasil proses).
    """

    @staticmethod
    def apply_brightness_contrast(image, brightness=0, contrast=0):
        """
        Mengubah brightness dan contrast gambar.
        Brightness: -100 to +100
        Contrast: -100 to +100
        """
        if image is None: return None
        
        # Penyesuaian contrast
        # Formula OpenCV yang umum: f(x) = alpha * x + beta
        # alpha = contrast control (1.0 - 3.0)
        # beta = brightness control (0 - 100)
        
        # Konversi ke float untuk kalkulasi yang lebih baik
        img_float = np.float64(image)
        
        # Normalisasi contrast agar lebih halus perubahannya
        # -100 -> 0.0, 0 -> 1.0, 100 -> 2.0
        alpha = (contrast + 100) / 100.0
        
        # Aplikasi Brightness dan Contrast
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=brightness)
        
        return adjusted

    @staticmethod
    def apply_hue_saturation_exposure(image, hue=0, saturation=0, exposure=0):
        """
        Mengatur Hue, Saturation, dan Exposure citra menggunakan ruang warna HSV.
        hue: -180 to 180 (derajat offset)
        saturation: -100 to 100 (persentase offset)
        exposure: -100 to 100 (persentase offset)
        """
        if image is None: return None
        
        # Konversi ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)
        
        # 1. Atur Hue (H: 0-179 dalam OpenCV)
        if hue != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + (hue / 2)) % 180
            
        # 2. Atur Saturation (S: 0-255)
        if saturation != 0:
            factor = (saturation + 100) / 100.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            
        # 3. Atur Exposure/Value (V: 0-255)
        if exposure != 0:
            factor = (exposure + 100) / 100.0
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
            
        # Kembalikan ke format uint8 dan konversi ke BGR
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return adjusted

    @staticmethod
    def apply_threshold(image, thresh_val):
        """
        Mengubah gambar menjadi biner (hitam-putih) berdasarkan nilai threshold (0-255).
        """
        if image is None: return None
        
        # Thresholding biasanya dilakukan pada citra grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        _, threshed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Kembalikan ke BGR agar bisa ditampilkan di GUI dengan fungsi yang sama
        if len(image.shape) == 3:
             threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)
             
        return threshed

    @staticmethod
    def to_grayscale(image):
        if image is None: return None
        if len(image.shape) == 2: return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=5):
        if image is None: return None
        if kernel_size % 2 == 0: kernel_size += 1 # Kernel harus ganjil
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_median_blur(image, kernel_size=5):
        if image is None: return None
        if kernel_size % 2 == 0: kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_edge_detection_canny(image, threshold1=100, threshold2=200):
        if image is None: return None
        # Canny edge detection
        edges = cv2.Canny(image, threshold1, threshold2)
        # Konversi ke 3 channel (BGR) untuk konsistensi penampil GUI
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def rotate_image(image, angle):
        """
        Rotasi citra menggunakan affine transform OpenCV
        """
        if image is None: return None
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Dapatkan matrix rotasi
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Hitung ukuran bounding box baru agar gambar tidak terpotong
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Sesuaikan matriks rotasi (translasi)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated

    @staticmethod
    def flip_image(image, mode_code):
        """
        mode_code: 1 = horizontal, 0 = vertical, -1 = both
        """
        if image is None: return None
        return cv2.flip(image, mode_code)

    @staticmethod
    def show_histogram(image, title="Histogram"):
        """
        Menampilkan histogram menggunakan Matplotlib.
        Dijalankan dengan cara yang tidak memblokir antarmuka jika memungkinkan,
        atau sebagai dialog popup modal.
        """
        if image is None: return
        
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.xlabel("Intensitas Piksel")
        plt.ylabel("Jumlah Piksel")
        
        if len(image.shape) == 3: # Warna BGR
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
        else: # Grayscale
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, color='black')
            plt.xlim([0, 256])
            
        plt.show() # Ini akan menampilkan window matplotlib terpisah

    @staticmethod
    def resize_image(image, width=None, height=None, interp=cv2.INTER_LINEAR):
        if image is None: return None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=interp)

    @staticmethod
    def crop_image(image, start_y, end_y, start_x, end_x):
        if image is None: return None
        return image[start_y:end_y, start_x:end_x]

    @staticmethod
    def split_rgb(image):
        """
        Splits image into R, G, B channels, but returns them as 3-channel images 
        so they can be displayed correctly.
        """
        if image is None or len(image.shape) != 3: return None
        b, g, r = cv2.split(image)
        zeros = np.zeros_like(b)
        img_b = cv2.merge((b, zeros, zeros))
        img_g = cv2.merge((zeros, g, zeros))
        img_r = cv2.merge((zeros, zeros, r))
        return img_r, img_g, img_b # Return R, G, B

    @staticmethod
    def color_segmentation(image, lower_hsv, upper_hsv):
        """
        Segmentasi warna menggunakan ruang HSV.
        Contoh lower_hsv = np.array([35, 14, 14])
        Contoh upper_hsv = np.array([75, 255, 255])
        """
        if image is None: return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        res = cv2.bitwise_and(image, image, mask=mask)
        return res
