import cv2
import numpy as np


class ImageProcessor:
    """
    Kelas utilitas untuk menangani seluruh operasi pengolahan citra digital.

    Semua metode bersifat statis (static methods) sehingga tidak memerlukan
    instansiasi kelas. Setiap metode menerima gambar dalam format NumPy array
    (format OpenCV BGR) dan mengembalikan gambar baru hasil proses tanpa
    mengubah gambar asli (non-destructive).

    Format Gambar:
        - Berwarna: numpy.ndarray dengan shape (H, W, 3) dalam urutan BGR
        - Grayscale: numpy.ndarray dengan shape (H, W) atau (H, W, 3)
        - Tipe data: numpy.uint8 (nilai piksel 0–255)
    """

    @staticmethod
    def apply_brightness_contrast(image, brightness=0, contrast=0):
        """
        Mengubah kecerahan (brightness) dan kontras (contrast) gambar.

        Menggunakan rumus transformasi linear OpenCV:
            output = alpha × input + beta

        di mana:
            - alpha (faktor kontras) = (contrast + 100) / 100
              → contrast = -100 menghasilkan alpha = 0.0 (gambar hitam total)
              → contrast =    0 menghasilkan alpha = 1.0 (tidak ada perubahan)
              → contrast = +100 menghasilkan alpha = 2.0 (kontras maksimum)
            - beta (offset kecerahan) = nilai brightness langsung

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            brightness (int): Nilai offset kecerahan. Rentang: -100 hingga +100.
            contrast (int): Nilai perubahan kontras. Rentang: -100 hingga +100.

        Returns:
            numpy.ndarray: Gambar hasil penyesuaian brightness/contrast,
                           atau None jika input tidak valid.
        """
        if image is None:
            return None

        # alpha mengontrol kontras, beta mengontrol kecerahan
        alpha = (contrast + 100) / 100.0
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=brightness)
        return adjusted

    @staticmethod
    def apply_hue_saturation_exposure(image, hue=0, saturation=0, exposure=0):
        """
        Mengatur Hue (warna), Saturation (kejenuhan), dan Exposure (kecerahan HSV)
        pada citra menggunakan ruang warna HSV.

        Pemrosesan dilakukan di ruang warna HSV untuk hasil yang lebih natural
        dibandingkan penyesuaian langsung di ruang BGR.

        Args:
            image (numpy.ndarray): Gambar input format BGR.
            hue (int): Pergeseran sudut warna dalam derajat. Rentang: -180 hingga +180.
                       (Dikonversi ke skala OpenCV 0–179 secara otomatis)
            saturation (int): Persentase perubahan kejenuhan. Rentang: -100 hingga +100.
                              0 = tidak berubah, -100 = abu-abu total, +100 = 2× lebih jenuh.
            exposure (int): Persentase perubahan kecerahan (Value channel HSV).
                            Rentang: -100 hingga +100.

        Returns:
            numpy.ndarray: Gambar hasil penyesuaian dalam format BGR,
                           atau None jika input tidak valid.
        """
        if image is None:
            return None

        # Konversi ke ruang warna HSV dan ubah ke float64 untuk presisi kalkulasi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float64)

        # 1. Geser Hue (H: rentang 0–179 dalam OpenCV, setara 0–358 derajat)
        if hue != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + (hue / 2)) % 180

        # 2. Skalakan Saturation (S: rentang 0–255)
        if saturation != 0:
            factor = (saturation + 100) / 100.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

        # 3. Skalakan Exposure/Value (V: rentang 0–255)
        if exposure != 0:
            factor = (exposure + 100) / 100.0
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)

        # Kembalikan ke uint8 dan konversi ke format BGR
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return adjusted

    @staticmethod
    def apply_threshold(image, thresh_val):
        """
        Mengonversi gambar menjadi citra biner (hitam-putih) menggunakan
        nilai ambang batas (threshold) tertentu.

        Setiap piksel dengan intensitas > thresh_val diset ke putih (255),
        sisanya diset ke hitam (0). Proses selalu dilakukan pada citra
        grayscale; jika input berwarna, akan dikonversi terlebih dahulu.

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            thresh_val (int): Nilai ambang batas. Rentang: 0 hingga 255.

        Returns:
            numpy.ndarray: Citra biner dalam format BGR (3 channel) agar
                           konsisten dengan pipeline GUI, atau None jika
                           input tidak valid.
        """
        if image is None:
            return None

        # Thresholding dilakukan pada citra grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        _, threshed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # Konversi kembali ke BGR agar konsisten dengan pipeline tampilan GUI
        if len(image.shape) == 3:
            threshed = cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR)

        return threshed

    @staticmethod
    def to_grayscale(image):
        """
        Mengonversi gambar berwarna ke representasi grayscale.

        Gambar dikembalikan dalam format 3-channel BGR (bukan 1-channel)
        agar tetap kompatibel dengan pipeline tampilan di GUI yang
        menggunakan Format_RGB888.

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.

        Returns:
            numpy.ndarray: Gambar grayscale dalam format BGR (3 channel),
                           atau None jika input tidak valid.
        """
        if image is None:
            return None
        # Jika sudah grayscale 1-channel, langsung kembalikan
        if len(image.shape) == 2:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Konversi kembali ke 3-channel agar konsisten dengan pipeline GUI
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=5):
        """
        Menerapkan filter Gaussian Blur untuk menghaluskan gambar dan
        mereduksi noise.

        Gaussian Blur menggunakan kernel berbentuk distribusi Gaussian
        yang memberikan transisi halus. Efektif untuk noise reduction
        sebelum deteksi tepi.

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            kernel_size (int): Ukuran kernel (harus ganjil). Default: 5.
                               Semakin besar nilai, semakin kuat efek blur.

        Returns:
            numpy.ndarray: Gambar hasil blurring, atau None jika input tidak valid.
        """
        if image is None:
            return None
        # Kernel harus bernilai ganjil
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_median_blur(image, kernel_size=5):
        """
        Menerapkan filter Median Blur untuk mereduksi noise sambil
        mempertahankan ketajaman tepi gambar.

        Median Blur menggantikan setiap piksel dengan nilai median
        dari piksel-piksel tetangganya. Sangat efektif untuk
        menghilangkan salt-and-pepper noise.

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            kernel_size (int): Ukuran kernel (harus ganjil). Default: 5.

        Returns:
            numpy.ndarray: Gambar hasil blurring, atau None jika input tidak valid.
        """
        if image is None:
            return None
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_edge_detection_canny(image, threshold1=100, threshold2=200):
        """
        Mendeteksi tepi (edge) pada gambar menggunakan algoritma Canny Edge Detection.

        Algoritma Canny melakukan beberapa tahap:
            1. Gaussian smoothing untuk mereduksi noise.
            2. Perhitungan gradien intensitas.
            3. Non-maximum suppression untuk menipiskan tepi.
            4. Double thresholding dan edge tracking by hysteresis.

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            threshold1 (int): Ambang batas bawah (hysteresis). Default: 100.
            threshold2 (int): Ambang batas atas (hysteresis). Default: 200.

        Returns:
            numpy.ndarray: Peta tepi dalam format BGR (3-channel) agar
                           kompatibel dengan pipeline GUI, atau None jika
                           input tidak valid.
        """
        if image is None:
            return None
        edges = cv2.Canny(image, threshold1, threshold2)
        # Konversi ke 3-channel BGR agar konsisten dengan pipeline tampilan GUI
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def rotate_image(image, angle):
        """
        Memutar gambar sebesar sudut tertentu menggunakan transformasi afin
        (affine transform) tanpa memotong konten gambar.

        Ukuran kanvas output disesuaikan secara otomatis agar seluruh
        gambar hasil rotasi tetap terlihat (tidak ada bagian yang terpotong).

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            angle (float): Sudut rotasi dalam derajat.
                           Nilai positif = berlawanan arah jarum jam.
                           Nilai negatif = searah jarum jam.

        Returns:
            numpy.ndarray: Gambar hasil rotasi, atau None jika input tidak valid.
        """
        if image is None:
            return None

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Dapatkan matriks rotasi 2×3
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Hitung dimensi bounding box baru agar gambar tidak terpotong
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1])
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))

        # Sesuaikan matriks dengan translasi ke pusat kanvas baru
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated

    @staticmethod
    def flip_image(image, mode_code):
        """
        Membalik (flip) gambar secara horizontal, vertikal, atau keduanya.

        Args:
            image (numpy.ndarray): Gambar input format BGR atau Grayscale.
            mode_code (int): Kode mode pembalikan:
                -  1 = flip horizontal (cermin kiri-kanan)
                -  0 = flip vertikal (cermin atas-bawah)
                - -1 = flip horizontal dan vertikal sekaligus

        Returns:
            numpy.ndarray: Gambar hasil flip, atau None jika input tidak valid.
        """
        if image is None:
            return None
        return cv2.flip(image, mode_code)

    @staticmethod
    def resize_image(image, width=None, height=None, interp=cv2.INTER_LINEAR):
        """
        Mengubah ukuran (resize) gambar dengan mempertahankan aspek rasio
        jika hanya satu dimensi yang diberikan.

        Args:
            image (numpy.ndarray): Gambar input.
            width (int, optional): Lebar target dalam piksel.
            height (int, optional): Tinggi target dalam piksel.
            interp: Metode interpolasi OpenCV. Default: cv2.INTER_LINEAR.

        Returns:
            numpy.ndarray: Gambar hasil resize, atau None jika input tidak valid.
                           Mengembalikan gambar asli jika width dan height keduanya None.
        """
        if image is None:
            return None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            ratio = height / float(h)
            dim = (int(w * ratio), height)
        else:
            ratio = width / float(w)
            dim = (width, int(h * ratio))
        return cv2.resize(image, dim, interpolation=interp)

    @staticmethod
    def crop_image(image, start_y, end_y, start_x, end_x):
        """
        Memotong (crop) gambar berdasarkan koordinat piksel yang ditentukan.

        Koordinat menggunakan sistem indeks baris-kolom NumPy (row-major):
            - start_y / end_y → batas atas dan bawah (sumbu vertikal)
            - start_x / end_x → batas kiri dan kanan (sumbu horizontal)

        Args:
            image (numpy.ndarray): Gambar input.
            start_y (int): Baris piksel awal (tepi atas area crop).
            end_y (int): Baris piksel akhir (tepi bawah area crop, eksklusif).
            start_x (int): Kolom piksel awal (tepi kiri area crop).
            end_x (int): Kolom piksel akhir (tepi kanan area crop, eksklusif).

        Returns:
            numpy.ndarray: Sub-gambar hasil crop, atau None jika input tidak valid.
        """
        if image is None:
            return None
        return image[start_y:end_y, start_x:end_x]

    @staticmethod
    def split_rgb(image):
        """
        Memisahkan gambar berwarna menjadi tiga gambar per channel warna
        (Merah, Hijau, Biru) dalam format 3-channel.

        Setiap gambar keluaran menampilkan satu channel warna dalam konteks
        warna aslinya (bukan grayscale), sehingga:
            - Channel R: gambar merah dengan G=0 dan B=0
            - Channel G: gambar hijau dengan R=0 dan B=0
            - Channel B: gambar biru dengan R=0 dan G=0

        Args:
            image (numpy.ndarray): Gambar input format BGR dengan 3 channel.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray] atau None:
                Tuple (img_red, img_green, img_blue) dalam format BGR.
                Mengembalikan None jika input tidak valid atau bukan 3-channel.
        """
        if image is None or len(image.shape) != 3:
            return None
        b, g, r = cv2.split(image)
        zeros = np.zeros_like(b)
        img_r = cv2.merge((zeros, zeros, r))   # Hanya channel Merah
        img_g = cv2.merge((zeros, g, zeros))   # Hanya channel Hijau
        img_b = cv2.merge((b, zeros, zeros))   # Hanya channel Biru
        return img_r, img_g, img_b

    @staticmethod
    def color_segmentation(image, lower_hsv, upper_hsv):
        """
        Melakukan segmentasi warna pada gambar berdasarkan rentang warna
        yang ditentukan dalam ruang warna HSV.

        Piksel yang berada di dalam rentang HSV yang ditentukan akan
        dipertahankan, sedangkan piksel di luar rentang akan diset ke hitam.
        Teknik ini berguna untuk mengisolasi objek berdasarkan warnanya.

        Args:
            image (numpy.ndarray): Gambar input format BGR.
            lower_hsv (numpy.ndarray): Batas bawah warna dalam format HSV.
                Contoh warna hijau: np.array([35, 50, 50])
            upper_hsv (numpy.ndarray): Batas atas warna dalam format HSV.
                Contoh warna hijau: np.array([85, 255, 255])

        Returns:
            numpy.ndarray: Gambar hasil segmentasi (piksel di luar rentang
                           berwarna hitam), atau None jika input tidak valid.
        """
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result
