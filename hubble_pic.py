# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io as sio

data_file = r'C:\Users\zyv57124\Documents\Documents\Galaxy_Zoo\ps58_4x6_Copy.png'

from io import BytesIO
from PIL import Image, ImageDraw

def extract_bits(color, bitmask):
    bitmask_len = len(bin(bitmask)[2:])
    extracted_bits = bin(color & bitmask)[2:]
    extracted_bits = '0' * (bitmask_len - len(extracted_bits)) + extracted_bits

    return extracted_bits

if __name__ == '__main__':
    img = Image.open(data_file)
    pixels = list(img.getdata())

    bits = ''
    for i in range(0, len(pixels), 1):
        r = pixels[i][0]
        g = pixels[i][1]
        b = pixels[i][2]

        if not (r <= 1 and g <= 1 and b <= 1): continue

        bits += extract_bits(r, 0x1)
        bits += extract_bits(g, 0x1)
        bits += extract_bits(b, 0x1)

    bits += '0' * (8 - len(bits) % 8)

    text = ''
    for i in range(0, len(bits), 8):
        text += chr(int(bits[i:i+8], 2))
    print (text)