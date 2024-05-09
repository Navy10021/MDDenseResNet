from PIL import Image
import numpy as np 
import os

def read_hex_file(file_path):
    valid_hex_chars = "0123456789abcdefABCDEF"
    hex_data = ""

    with open(file_path, 'r') as file:
        for line in file:
            # 1. extract only valid hex chars
            filtered_line = ''.join(filter(lambda x : x in valid_hex_chars, line.strip()))
            # 2. extract valid hex chars or convert to '0'
            #filtered_line = ''.join(char if char in valid_hex_chars else '0' for char in line.strip())

            hex_data += filtered_line
        return hex_data


def hex_to_image(hex_data, width):
    # Convert hexadecimal data to byte array
    bytes_data = bytes.fromhex(hex_data) 
    
    # Padding with zero('x00')
    padding = (width - len(bytes_data) % width) % width
    bytes_data += b'\x00' * padding 

    # Calculate the height of the image (so all data can fit)
    height = len(bytes_data) // width
    
    # Convert bytes data to array 
    arr_data = np.frombuffer(bytes_data, dtype=np.uint8)
    arr_data = arr_data.reshape((height, width))
    
    # generate image
    img = Image.fromarray(arr_data, 'L')
    return img 


def hex_to_rgb_image(hex_data, width):
    bytes_data = bytes.fromhex(hex_data)
    
    # Calculate padding to ensure the length is a multiple of (width * 3)
    padding = (width * 3 - len(bytes_data) % (width * 3)) % (width * 3)
    bytes_data += b'\x00' * padding
    
    # Calculate height based on the total length divided by (width * 3)
    height = len(bytes_data) // (width * 3)

    #print("Width:", width)
    #print("Height:", height)
    #print("Array size:", len(bytes_data))

    # Reshape array to (height, width, 3) for RGB image
    array_data = np.frombuffer(bytes_data, dtype=np.uint8)
    array_data = array_data.reshape((height, width, 3))

    # Generate RGB image
    image = Image.fromarray(array_data, 'RGB')
    return image


def check_images(folder_path):
    error_images = []

    # 이미지 폴더 내의 각 폴더 순회
    for subdir, dirs, files in os.walk(folder_path):
        # 각 폴더 내의 이미지 파일 순회
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                # 이미지 파일 열기 시도
                img = Image.open(file_path)
                img.verify()  # 이미지 유효성 확인
            except (IOError, SyntaxError) as e:
                # 오류 발생 시 해당 파일을 오류 이미지 목록에 추가
                error_images.append(file_path)
    
    return error_images
"""
folder_path = "path_to_your_image_folder"
error_images = check_images(folder_path)
    
if error_images:
    print("Error images found:")
    for img_path in error_images:
        print(img_path)
else:
    print("No error images found.")
"""