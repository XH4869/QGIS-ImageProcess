import cv2
import numpy as np

class Merge:
	def __init__(self):
		pass

	def graytograyimg(self, img):
		grayimg = img * 1
		width = img.shape[0]
		height = img.shape[1]
		for i in range(width):
			for j in range(height):
				grayimg[i, j] = 0.299 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 2]

		return grayimg

	def graytoHSgry(self, grayimg, HSVimg):
		H, S, V = cv2.split(HSVimg)
		rows, cols = V.shape
		for i in range(rows):
			for j in range(cols):
				V[i, j] = grayimg[i][j][0]

		newimg = cv2.merge([H, S, V])
		newimg = np.uint8(newimg)
		return newimg
