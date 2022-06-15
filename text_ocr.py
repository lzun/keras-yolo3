import pytesseract
from PIL import Image, ImageOps
import os

BASE_DIR = '/Users/luisnzunigamorales/Documents/GitHub/keras-yolo3/'
BASE_SUBIMAGE_DIR = '/Users/luisnzunigamorales/Documents/GitHub/keras-yolo3/subimages/'
W_BASE_DIR = 'C:\\Users\\green\\Documents\\GitHub\\keras-yolo3\\'
W_BASE_SUBIMAGE_DIR = 'C:\\Users\\green\\Documents\\GitHub\\keras-yolo3\\subimages\\'

def get_image_dir(folder_dir, file_type = 'jpg'):
    """Detect all images in the specified folder. Returns a list with the string paths."""

    file_list = []

    for image in os.listdir(folder_dir):
        # check if the image ends with jpg
        if (image.endswith('.' + file_type)):
            file_list.append(image)

    return file_list

def text_from_image(img_file, lang = 'spa'):

    # extrae el texto de la imagen con el OCR Tesseract
    return pytesseract.image_to_string(ImageOps.grayscale(Image.open(img_file)), lang=lang)

def main(dir_path = W_BASE_SUBIMAGE_DIR):

    # iteramos sobre las carpetas de la dirección base
    for filename in os.listdir(dir_path):
        f = os.path.join(dir_path, filename)
        if (os.path.isdir(f)):
            # get file list of image paths
            file_list = get_image_dir(f)
            for path in file_list:
                text = text_from_image(os.path.join(f, path))
                os.chdir(f)
                with open(path[:-4]+'.txt','w') as wfile:
                    wfile.write(text)
                os.chdir(W_BASE_SUBIMAGE_DIR)
        else:
            pass

if __name__ == '__main__':
    main()