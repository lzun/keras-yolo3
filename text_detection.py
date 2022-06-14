import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

import os
import numpy as np

BASE_DETECT_DIR = 'images/'
BASE_SAVE_DIR = 'box_images/'

def get_image_dir(folder_dir, file_type = 'jpg'):
    """Detect all images in the specified folder. Returns a list with the string paths."""

    file_list = []

    for image in os.listdir(folder_dir):
        # check if the image ends with jpg
        if (image.endswith('.' + file_type)):
            file_list.append(image)

    return file_list

def detect_text(yolo, dir_path = BASE_DETECT_DIR):

    # get file list of image paths
    file_list = get_image_dir(dir_path)

    for path in file_list:
        print(dir_path + path)
        # abrimos la imagen
        img = Image.open(dir_path + path)
        # la pasamos a YOLO para que detecte las cajas de texto
        r_image, out_boxes, out_scores, out_classes = yolo.detect_image(img)

        # guardamos la imagen con las cajas dibujadas
        r_image.save(BASE_SAVE_DIR + path)

        # guardamos el archivo con las coordenadas de las cajas
        with open('bbox_file_' + path[:-4] + '.txt', 'a') as f:
            for i in range(0,len(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(img.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(img.size[0], np.floor(right + 0.5).astype('int32'))
        
                f.write(str(left)+','+str(top)+','+str(right)+','+str(bottom)+','+str(out_scores[i])+','+str(out_classes[0])+'\n')

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

if __name__ == '__main__':

    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_text(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")