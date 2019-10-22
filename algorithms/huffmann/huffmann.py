from collections import Counter
import argparse
import heapq
import os

class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __gt__(self, other):
        if(other == None):
            return -1
        if(not isinstance(other, HeapNode)):
            return -1
        return self.freq > other.freq

class HuffmanCoding:

    def __init__(self, path, limit=None):
        self.path = path
        self.heap = []
        self.codes = {}
        self.reverse_codes = {}
        with open(self.path, 'r') as f:
            if limit:
                self.text = f.read()[:limit].rstrip()
            else:
                self.text = f.read().rstrip()
        self.frequency = self.make_frequency_dict(self.text)
        self.make_heap(self.frequency)
        self.merge_nodes()
        self.make_codes()

    # functions for compression:

    def make_frequency_dict(self, text):
        frequency = Counter()
        for c in text: frequency[c] += 1
        return frequency

    def make_heap(self, freq_dict):
        for l, freq in freq_dict.items():
            node = HeapNode(l, freq)
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if root is None: return
        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_codes[current_code] = root.char
            return
        self.make_codes_helper(root.left, current_code + '0')
        self.make_codes_helper(root.right, current_code + '1')

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ''
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ''
        for c in text:
            encoded_text += self.codes[c]
        return encoded_text

    # make encoded text a multiple of 8 (bytes)
    # append the appropriate number of 0s at the end
    # & prepend this information as an 8 bits string
    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8    # complement of the modulo
        encoded_text += '0' * extra_padding          # append modulo_complement zeros at the end
        padded_info = '{:08b}'.format(extra_padding) # format: 0 to keep zeros, b for binary
        encoded_text = padded_info + encoded_text    # prepend modulo in the form of padding
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if len(padded_encoded_text) % 8 != 0:
            print('Encoded text not padded properly')
            exit(0)
        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b

    def compress(self, limit=None):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + '.bin'

        with open(output_path, 'wb') as output:
            encoded_text = self.get_encoded_text(self.text)
            padded_encoded_text = self.pad_encoded_text(encoded_text)
            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))

        print('Compressed to:', filename+'.bin')
        return output_path

    # functions for decompression

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]               # retrieve info
        extra_padding = int(padded_info, 2)                 # convert back to int
        padded_encoded_text = padded_encoded_text[8:]       # text without info
        encoded_text = padded_encoded_text[:-extra_padding] # text without end padding
        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ''
        decoded_text = ''

        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_codes:
                character = self.reverse_codes[current_code]
                decoded_text += character
                current_code = ''

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + '_decompressed' + '.txt'

        with open(input_path, 'rb') as f, open(output_path, 'w') as output:
            bit_string = ''
            index = 0
            byte = f.read(1)
            while byte:
                index += 1
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = f.read(1)

            encoded_text = self.remove_padding(bit_string)
            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print('Decompressed to:', output_path)
        return output_path

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='''
            Huffmann encoding and decoding. Imported & modified from:
            http://bhrigu.me/blog/2017/01/17/huffman-coding-python-implementation/''')
    parser.add_argument('--compress', '-c', 
                        dest='enc', action='store_true', default=True, 
                        help='Compress the file into binary format.')
    parser.add_argument('--decompress', '-d', dest='enc', action='store_false',
                        help='Decompress the binary file back to plain text.')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit the amount of characters to encode from the file (for testing).')
    parser.add_argument('--path', '-p', default='fw.txt', 
                        help='The source file path.')
    parser.add_argument('--bin_path', '-b', default='fw.bin', 
                        help='The binary file path.')

    args = parser.parse_args()

    h = HuffmanCoding(args.path)

    if args.enc:
        h.compress(limit=args.limit)
    else:
        h.decompress(args.bin_path)

