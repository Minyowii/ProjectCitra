import cv2
import numpy as np
import threading

class HumanDetector:
    """
    Kelas untuk menangani deteksi manusia menggunakan model MobileNetV2
    melalui TensorFlow / TensorFlow Hub.
    """
    def __init__(self):
        self.detector = None
        self.is_loading = False
        self.is_ready = False

    def load_model(self, callback=None):
        """
        Memuat model dari TF Hub secara asynchronous agar tidak memblokir GUI.
        """
        if self.is_ready or self.is_loading:
            if callback: callback(True, "")
            return

        self.is_loading = True
        
        def _load():
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                
                # Menggunakan MobileNet V2 SSD dari TF Hub
                module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
                self.detector = hub.load(module_handle).signatures['default']
                self.is_ready = True
                self.is_loading = False
                if callback: callback(True, "")
            except ImportError:
                print("Error: Library tensorflow atau tensorflow_hub belum terinstall.")
                self.is_loading = False
                if callback: callback(False, "Library tensorflow atau tensorflow_hub belum terinstall.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.is_loading = False
                if callback: callback(False, str(e))

        threading.Thread(target=_load).start()

    def detect(self, image):
        """
        Mendeteksi manusia dan menggambar Bounding Box beserta Confidence Score.
        """
        if not self.is_ready or image is None:
            return image
            
        import tensorflow as tf
        
        # Konversi BGR OpenCV ke RGB karena model dilatih dengan RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Konversi ke tensor float32 dan tambahkan dimensi batch
        converted_img = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]
        
        # Inferensi
        result = self.detector(converted_img)
        result = {key: value.numpy() for key, value in result.items()}
        
        output_image = image.copy()
        h, w, _ = output_image.shape
        
        human_detected = False
        
        for i in range(len(result["detection_scores"])):
            score = result["detection_scores"][i]
            class_name = result["detection_class_entities"][i].decode("ascii")
            
            # Filter untuk kelas yang merepresentasikan manusia dengan confidence >= 15%
            valid_classes = ["Person", "Woman", "Man", "Girl", "Boy", "Human body", "Human face", "Human"]
            if any(vc in class_name for vc in valid_classes) and score >= 0.15:
                human_detected = True
                ymin, xmin, ymax, xmax = result["detection_boxes"][i]
                
                # Konversi koordinat relatif ke piksel absolut
                ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
                
                # Gambar kotak (hijau)
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Tulis label confidence score
                text = f"{class_name}: {score * 100:.1f}%"
                
                # Latar belakang untuk teks agar lebih mudah dibaca
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output_image, (xmin, ymin - text_height - 10), (xmin + text_width, ymin), (0, 255, 0), cv2.FILLED)
                cv2.putText(output_image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        return output_image, human_detected
