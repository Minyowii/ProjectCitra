import numpy as np
import heapq
from collections import Counter

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, nxt):
        return self.freq < nxt.freq

class CompressionSimulator:
    """
    Modul untuk mensimulasikan proses kompresi citra (akademik).
    Melibatkan tahapan Kuantisasi dan pengkodean Huffman.
    """
    
    @staticmethod
    def quantize(image, levels=32):
        """
        Kuantisasi citra ke jumlah level diskrit tertentu.
        Tujuannya mengurangi variasi nilai piksel agar probabilitas simbol tertentu naik,
        sehingga kompresi Huffman lebih efektif.
        """
        if image is None: return None
        # Normalisasi ke rentang yang lebih kecil
        factor = 256 / levels
        quantized = np.uint8(np.floor(image / factor) * factor)
        return quantized

    @staticmethod
    def simulate_huffman(image):
        """
        Mensimulasikan Huffman coding:
        1. Menghitung frekuensi kemunculan setiap nilai piksel.
        2. Membangun Huffman Tree.
        3. Membuat dictionary kodeword.
        4. Menghitung total bit yang dibutuhkan.
        """
        if image is None: return None
        
        # Ratakan (flatten) image 3D/2D menjadi array 1D
        flat_img = image.flatten()
        total_pixels = len(flat_img)
        
        # Asumsi 8 bit per nilai channel/piksel
        original_bits = total_pixels * 8
        
        # 1. Hitung frekuensi
        freq_dict = Counter(flat_img)
        
        # 2. Bangun pohon Huffman
        nodes = []
        for symbol, freq in freq_dict.items():
            heapq.heappush(nodes, Node(freq, symbol))
            
        if len(nodes) == 0:
            return None
            
        if len(nodes) == 1:
            return {
                "original_bytes": original_bits / 8,
                "compressed_bytes": original_bits / 8,
                "ratio": 1.0,
                "space_saving": 0.0
            }
            
        while len(nodes) > 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            
            left.huff = 0
            right.huff = 1
            
            # Buat node gabungan
            newNode = Node(left.freq + right.freq, "INTERNAL", left, right)
            heapq.heappush(nodes, newNode)
            
        # 3. Traverse tree untuk mendapatkan kodeword tiap simbol
        huffman_dict = {}
        def generate_codes(node, val=''):
            newVal = val + str(node.huff)
            if node.left:
                generate_codes(node.left, newVal)
            if node.right:
                generate_codes(node.right, newVal)
            if not node.left and not node.right:
                huffman_dict[node.symbol] = newVal

        generate_codes(nodes[0])
        
        # 4. Hitung ukuran file terkompresi
        # Jumlah bit = jumlah kemunculan * panjang kodeword Huffman
        compressed_bits = sum(freq_dict[symbol] * len(code) for symbol, code in huffman_dict.items())
        
        original_bytes = original_bits / 8
        compressed_bytes = compressed_bits / 8
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1
        space_saving = 1 - (compressed_bytes / original_bytes) if original_bytes > 0 else 0
        
        return {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "ratio": ratio,
            "space_saving": space_saving * 100
        }
