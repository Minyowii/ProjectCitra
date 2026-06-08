import cv2
import numpy as np
import threading


class HumanDetector:
    """
    Kelas untuk menangani deteksi manusia pada citra menggunakan model
    MobileNet SSD (Caffe) yang dijalankan melalui OpenCV DNN.

    Model dijalankan secara asinkron saat pertama kali diminta agar
    antarmuka pengguna tidak mengalami pembekuan (GUI freeze).

    Atribut:
        detector: Objek net dari cv2.dnn.
        is_loading (bool): True saat model sedang diunduh/dimuat.
        is_ready (bool): True saat model sudah siap digunakan.
    """

    # Ambang batas confidence minimum (15%)
    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self):
        self.detector = None
        self.is_loading = False
        self.is_ready = False

    def load_model(self, callback=None):
        """
        Memuat model deteksi objek secara asinkron menggunakan thread terpisah
        dan mengunduh file model secara otomatis jika belum tersedia di lokal.

        Args:
            callback (callable, optional): Fungsi yang dipanggil setelah
                proses selesai dengan signature callback(success: bool, error_msg: str).
        """
        if self.is_ready or self.is_loading:
            if callback:
                callback(True, "")
            return

        self.is_loading = True

        def _load():
            try:
                import os
                import requests

                # Lokasi file model relatif terhadap file ml_module.py
                base_dir = os.path.dirname(os.path.abspath(__file__))
                proto_path = os.path.join(base_dir, "deploy.prototxt")
                model_path = os.path.join(base_dir, "mobilenet_iter_73000.caffemodel")

                # URL download model MobileNet SSD Caffe
                proto_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
                model_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"

                # Unduh file prototxt jika belum ada
                if not os.path.exists(proto_path):
                    print("[HumanDetector] Mengunduh prototxt...")
                    r = requests.get(proto_url, timeout=15)
                    r.raise_for_status()
                    with open(proto_path, "wb") as f:
                        f.write(r.content)
                    print("[HumanDetector] Prototxt selesai diunduh.")

                # Unduh file caffemodel jika belum ada
                if not os.path.exists(model_path):
                    print("[HumanDetector] Mengunduh caffemodel...")
                    r = requests.get(model_url, timeout=60)
                    r.raise_for_status()
                    with open(model_path, "wb") as f:
                        f.write(r.content)
                    print("[HumanDetector] Caffemodel selesai diunduh.")

                # Muat model menggunakan OpenCV DNN
                print("[HumanDetector] Memuat model ke OpenCV DNN...")
                self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self.is_ready = True
                self.is_loading = False
                print("[HumanDetector] Model berhasil dimuat.")
                
                if callback:
                    callback(True, "")

            except Exception as e:
                err_msg = str(e)
                print(f"[HumanDetector] Gagal memuat model: {err_msg}")
                self.is_loading = False
                if callback:
                    callback(False, err_msg)

        threading.Thread(target=_load, daemon=True).start()

    def detect(self, image):
        """
        Mendeteksi keberadaan manusia pada gambar dan menggambar bounding box
        beserta label confidence score pada setiap deteksi yang valid.

        Args:
            image (numpy.ndarray): Gambar input dalam format BGR (OpenCV).

        Returns:
            tuple[numpy.ndarray, bool]:
                - Gambar hasil anotasi (sama seperti input jika tidak ada deteksi).
                - True jika minimal satu manusia terdeteksi, False jika tidak.
        """
        # Kembalikan gambar asli dan False jika model belum siap atau gambar kosong
        if not self.is_ready or image is None:
            return image, False

        h, w, _ = image.shape
        output_image = image.copy()
        human_detected = False

        # Persiapkan blob untuk OpenCV DNN MobileNet SSD
        # Menggunakan ukuran 300x300, faktor skala 1/127.5 (0.007843), dan mean 127.5
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.detector.setInput(blob)
        detections = self.detector.forward()

        # Detections shape: [1, 1, N, 7]
        # Format deteksi: [batch, class_id, score, xmin, ymin, xmax, ymax]
        for i in range(detections.shape[2]):
            score = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            # Class ID 15 mewakili objek "person" (manusia) di MobileNet SSD
            if class_id == 15 and score >= self.CONFIDENCE_THRESHOLD:
                human_detected = True

                # Ambil koordinat relatif [0..1] dan kalibrasi ke piksel gambar asli
                # Dibatasi dengan clip antara 0 dan 1 agar bbox tidak keluar batas kanvas
                xmin = int(max(0.0, min(1.0, detections[0, 0, i, 3])) * w)
                ymin = int(max(0.0, min(1.0, detections[0, 0, i, 4])) * h)
                xmax = int(max(0.0, min(1.0, detections[0, 0, i, 5])) * w)
                ymax = int(max(0.0, min(1.0, detections[0, 0, i, 6])) * h)

                # Gambar bounding box berwarna hijau
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Buat label teks dengan confidence score
                label = f"Person: {score * 100:.1f}%"

                # Hitung ukuran teks untuk latar belakang label
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )

                # Gambar latar belakang label (kotak hijau solid) di atas bbox
                y_label_start = max(text_h + 10, ymin)
                cv2.rectangle(
                    output_image,
                    (xmin, y_label_start - text_h - 10),
                    (xmin + text_w, y_label_start),
                    (0, 255, 0),
                    cv2.FILLED
                )

                # Tulis teks label (hitam di atas latar hijau)
                cv2.putText(
                    output_image, label,
                    (xmin, y_label_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )

        return output_image, human_detected
