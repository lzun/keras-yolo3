import cv2

BBOX_BASE_DIR = 'bbox_files_txt/'

# leer imagen
img = cv2.imread('images/1_r.jpg')

# cargar las cajas de texto
i = 1

with open (BBOX_BASE_DIR + 'bbox_file_' + str(i) + '.txt', 'r') as bbox_file:
	for line in bbox_file:
		coords = line.split(',')
		print(coords)
		# crop_img = img[int(coords[0]):int(coords[3]), int(coords[1]):int(coords[2])]
		crop_img = img[95:199,0:239]
		cv2.imshow("cropped", crop_img)
		cv2.waitKey(0)