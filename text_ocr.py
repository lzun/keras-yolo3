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

def keras_test():
    # https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/keras_ocr_custom.ipynb#scrollTo=-OQIMmIp-uwn
    
    custom_images = []

    pipeline = keras_ocr.pipeline.Pipeline()
    
    for filename in os.listdir('ocr_test'):
        print(os.path.join(W_BASE_DIR, filename))
        custom_images.append(os.path.join(W_BASE_DIR+'ocr_test\\', filename))

    images = [keras_ocr.tools.read(path) for path in custom_images]

    predictions = pipeline.recognize(images)

    # fig, axs = plt.subplots(nrows=len(images), figsize=(10, 10))
    # if(len(custom_images) == 1):
    #     for image, prediction in zip(images, predictions):
    #         keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=axs)
    # else:
    #     for ax, image, prediction in zip(axs, images, predictions):
    #         keras_ocr.tools.drawAnnotations(image=image, predictions=prediction, ax=ax)

    # plt.show()
    line = []

    for word, array in predictions[0]:
        line.append((array, word+' '))
    print(keras_ocr.tools.combine_line(line))

    # with open('results.txt', 'a+') as f:
    #     for idx, prediction in enumerate(predictions):
    #         if(idx != 0):
    #             print("\n")
    #             f.write("\n\n")
    #         print("Results for the file: " + os.path.basename(custom_images[idx]))
    #         f.write("Results for the file: " + os.path.basename(custom_images[idx]) + ":\n\n")
    #         for word, array in prediction:
    #             if word == "\n":
    #                 print("\n")
    #                 f.write("\n")
    #             else:
    #                 print(word,  end = ' ')
    #                 f.write(word + " ")

def text_from_image(img_file, lang = 'spa'):

    # extrae el texto de la imagen con el OCR Tesseract
    # return pytesseract.image_to_string(Image.open(img_file), lang=lang)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize(images)

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
    keras_test()