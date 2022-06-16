import cv2
import os

BBOX_BASE_DIR = 'bbox_files_txt/'
BASE_IMAGE_DIR = 'images_processed/'
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

def main(dir_path = BASE_IMAGE_DIR):

	# get file list of image paths
	file_list = get_image_dir(dir_path)

	for path in file_list:

		# se carga la imagen para detectar las cajas
		img = cv2.imread(dir_path + path)

		# se crea un directorio nuevo para almacenar las nuevas imagenes con el texto
		os.mkdir(os.path.join(W_BASE_SUBIMAGE_DIR + 'canelo_caption_'+ path[:-4]))

		# se carga el archivo de texto con las coordenadas
		with open(BBOX_BASE_DIR + 'bbox_file_' + path[:-5] + 'r.txt', 'r') as bbox_file: # quitar 'r' y path[:-4]
			img_counter = 1
			# nos cambiamos al nuevo directorio creado anteriormente
			os.chdir(os.path.join(W_BASE_SUBIMAGE_DIR + 'canelo_caption_'+ path[:-4]))
			# iteramos sobre cada caja detectada
			for line in bbox_file:
				coords = line.split(',')
				crop_img = img[int(coords[1]):int(coords[3]), int(coords[0]):int(coords[2])]
				# crop_img = img[95:199,0:239]
				cv2.imwrite('img_' + str(img_counter) + '.jpg', crop_img)
				img_counter += 1

			# regresamos a la carpeta base
			os.chdir(W_BASE_DIR)

if __name__ == '__main__':
	
	main()