import numpy as np
import heapq
from collections import Counter


class Node:
    """
    Merepresentasikan satu simpul (node) pada Pohon Huffman (Huffman Tree).

    Setiap simpul menyimpan frekuensi kemunculan simbol dan referensi ke
    anak kiri/kanan. Simpul daun (leaf) memiliki nilai simbol aktual,
    sedangkan simpul internal diberi label 'INTERNAL'.

    Atribut:
        freq (int): Frekuensi kemunculan simbol.
        symbol: Nilai simbol piksel (0–255) atau 'INTERNAL' untuk simpul gabungan.
        left (Node): Anak kiri pohon Huffman.
        right (Node): Anak kanan pohon Huffman.
        huff (str): Bit kode Huffman ('0' atau '1') untuk simpul ini.
    """

    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, nxt):
        """Membandingkan dua node berdasarkan frekuensi (untuk priority queue)."""
        return self.freq < nxt.freq


class CompressionSimulator:
    """
    Modul untuk mensimulasikan proses kompresi citra secara akademik.

    Alur kerja:
        1. Kuantisasi: Mengurangi variasi nilai piksel ke level tertentu.
        2. Huffman Coding: Membangun pohon dan menghitung panjang kode optimal.
        3. Pelaporan: Menampilkan statistik kompresi (rasio, penghematan ruang).

    Catatan:
        Modul ini merupakan simulasi teoritis — tidak menghasilkan file
        terkompresi yang sebenarnya, melainkan menghitung ukuran optimal
        berdasarkan teori Huffman coding.
    """

    @staticmethod
    def quantize(image, levels=32):
        """
        Melakukan kuantisasi (quantization) pada citra ke sejumlah level diskrit.

        Kuantisasi mengurangi jumlah nilai piksel yang unik sehingga distribusi
        probabilitas menjadi lebih terpusat. Hal ini meningkatkan efisiensi
        kompresi Huffman karena lebih banyak simbol yang memiliki probabilitas tinggi.

        Args:
            image (numpy.ndarray): Citra input (grayscale atau BGR).
            levels (int): Jumlah level kuantisasi. Default 32.
                          Semakin kecil nilainya, semakin tinggi kompresi
                          namun semakin rendah kualitas visual.

        Returns:
            numpy.ndarray: Citra yang sudah dikuantisasi dalam format uint8,
                           atau None jika input tidak valid.
        """
        if image is None:
            return None

        factor = 256 / levels
        quantized = np.uint8(np.floor(image / factor) * factor)
        return quantized

    @staticmethod
    def simulate_huffman(image):
        """
        Mensimulasikan proses Huffman Coding pada data piksel citra.

        Tahapan:
            1. Ratakan citra menjadi array 1D.
            2. Hitung frekuensi kemunculan setiap nilai piksel.
            3. Bangun Pohon Huffman menggunakan priority queue (min-heap).
            4. Generate kodeword (bit string) untuk setiap simbol.
            5. Hitung total bit yang dibutuhkan dan bandingkan dengan ukuran asli.

        Args:
            image (numpy.ndarray): Citra input, idealnya sudah dikuantisasi
                                   menggunakan `quantize()` terlebih dahulu.

        Returns:
            dict atau None: Dictionary berisi statistik kompresi dengan kunci:
                - 'original_bytes' (float): Ukuran data asli dalam bytes.
                - 'compressed_bytes' (float): Ukuran data terkompresi (Huffman) dalam bytes.
                - 'ratio' (float): Rasio kompresi (original / compressed).
                - 'space_saving' (float): Persentase penghematan ruang (0–100).
            Mengembalikan None jika input tidak valid atau tidak ada data.
        """
        if image is None:
            return None

        # Ratakan citra 2D/3D menjadi array 1D untuk analisis distribusi piksel
        flat_img = image.flatten()
        total_pixels = len(flat_img)

        # Ukuran asli: asumsikan 8 bit per nilai piksel (tanpa kompresi)
        original_bits = total_pixels * 8

        # Langkah 1: Hitung frekuensi setiap simbol piksel
        freq_dict = Counter(flat_img)

        # Langkah 2: Bangun priority queue dari simpul-simpul daun
        nodes = []
        for symbol, freq in freq_dict.items():
            heapq.heappush(nodes, Node(freq, symbol))

        if len(nodes) == 0:
            return None

        # Kasus khusus: hanya ada satu simbol unik
        if len(nodes) == 1:
            return {
                "original_bytes": original_bits / 8,
                "compressed_bytes": original_bits / 8,
                "ratio": 1.0,
                "space_saving": 0.0
            }

        # Langkah 3: Bangun Pohon Huffman dengan menggabungkan dua simpul
        # berfrekuensi terendah secara berulang
        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)

            left.huff = 0
            right.huff = 1

            # Buat simpul gabungan (internal node) dengan frekuensi total
            merged_node = Node(left.freq + right.freq, "INTERNAL", left, right)
            heapq.heappush(nodes, merged_node)

        # Langkah 4: Traverse pohon secara rekursif untuk generate kodeword
        huffman_dict = {}

        def generate_codes(node, current_code=''):
            """Rekursif: kumpulkan kodeword untuk setiap simpul daun."""
            new_code = current_code + str(node.huff)
            if node.left:
                generate_codes(node.left, new_code)
            if node.right:
                generate_codes(node.right, new_code)
            # Simpul daun: simpan kodeword final
            if not node.left and not node.right:
                huffman_dict[node.symbol] = new_code

        generate_codes(nodes[0])

        # Langkah 5: Hitung total bit terkompresi
        # = Σ (frekuensi_simbol × panjang_kodeword_huffman)
        compressed_bits = sum(
            freq_dict[symbol] * len(code)
            for symbol, code in huffman_dict.items()
        )

        original_bytes = original_bits / 8
        compressed_bytes = compressed_bits / 8
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        space_saving = (1 - (compressed_bytes / original_bytes)) * 100 if original_bytes > 0 else 0.0

        return {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": ratio,
            "space_saving": space_saving
        }
