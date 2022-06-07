import numpy as np
import cv2

#otsu 演算法
def otsu(img_gray):
  max_g = 0
  suitable_th = 0
  #灰階範圍 0 - 255
  for threshold in range(0,256):
    #判斷是否前景像素
    bin_img = img_gray > threshold
    #判斷是否背景像素
    bin_img_inv = img_gray <= threshold
    fore_pix = np.sum(bin_img)
    back_pix = np.sum(bin_img_inv)
    if 0 == fore_pix:
      break
    if 0 == back_pix:
      continue
    w0 = float(fore_pix) / img_gray.size
    u0 = float(np.sum(img_gray * bin_img)) / fore_pix
    w1 = float(back_pix) / img_gray.size
    u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
    # intra-class variance
    g = w0 * w1 * (u0 - u1) * (u0 - u1)
    if g > max_g:
      max_g = g
      suitable_th = threshold
  return suitable_th

#Binarize 二值化
def binarize(img,thresh,maxval,type_f=0):
  #輸入圖片img、閥值thresh、填充色maxval、閥值類型type
  #閥值類型
  #閥值，小於閥值得像素點，大於閥值像素點
  #0,置0，置填充色
  #1,置填充色,置0
  #2,保持原色,置閥值色
  #3,置0,保持原色
  #4,保持原色,置0
	#5,置0，置填充色，thresh改為otsu結果
  if type_f == 5:
    thresh = otsu(img)
  new_img = img.copy()
  for i in range(len(img)):
    for j in range(len(img[i])):
      if type_f == 1:
        min_val,max_val = maxval,0
      elif type_f == 2:
        min_val,max_val = img[i][j],thresh
      elif type_f == 3:
        min_val,max_val = 0,img[i][j]
      elif type_f == 4:
        min_val,max_val = img[i][j],0
      else: #type_f == 0
        min_val,max_val = 0,maxval
      new_img[i][j] = max_val if img[i][j] > thresh else min_val
  return new_img


image = cv2.imread("test.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

res = binarize(image_gray, 50, 255, 5)
cv2.imshow("image",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
