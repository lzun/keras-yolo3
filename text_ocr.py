import pytesseract
import keras_ocr
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

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

def keras_test(path):
    # https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/keras_ocr_custom.ipynb#scrollTo=-OQIMmIp-uwn

    # iniciamos el pipeline para el OCR
    pipeline = keras_ocr.pipeline.Pipeline()
    
    # realizamos la prediccion con el motor
    predictions = pipeline.recognize([keras_ocr.tools.read(path)])
    print(predictions)
    # intentamos darle sentido al texto
    line = []

    for word, array in predictions[0]:
        line.append((array, word+' '))

    print(keras_ocr.tools.combine_line(line))

    return keras_ocr.tools.combine_line(line)

def text_from_image(img_file, lang = 'spa'):

    # extrae el texto de la imagen con el OCR Tesseract
    # return pytesseract.image_to_string(Image.open(img_file), lang=lang)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)

def main(dir_path = W_BASE_SUBIMAGE_DIR):

    # iniciamos el pipeline para el OCR
    pipeline = keras_ocr.pipeline.Pipeline()

    # iteramos sobre las carpetas de la dirección base
    for filename in os.listdir(dir_path):
        f = os.path.join(dir_path, filename)
        # si el path es un directorio
        if (os.path.isdir(f)):
            # get file list of image paths
            file_list = get_image_dir(f)
            for path in file_list:
                print(f+path)
                # realizamos la prediccion con el motor
                predictions = pipeline.recognize([keras_ocr.tools.read(os.path.join(f, path))])
                # line = []
                # for word, array in predictions[0]:
                #     line.append((array, word+' '))
                try:  
                    text = keras_ocr.tools.combine_line([(array, word+' ') for word, array in predictions[0]])
                    print(text[1])
                    os.chdir(f)
                    with open(path[:-4]+'.txt','w') as wfile:
                        wfile.write(text[1])
                except:
                    pass
                os.chdir(W_BASE_SUBIMAGE_DIR)
        else:
            pass

if __name__ == '__main__':
    main()