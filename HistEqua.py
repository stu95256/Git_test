import numpy as np
import cv2

#Histogram Equalize Morphology 直方圖均衡化
def HistEqua(img):
  #圖片多維陣列轉一維，並計算數據集的直方圖
  hist,bins = np.histogram(img.ravel(),256,[0,255])
  #將每一個灰階級的次數累加，變成累積分布函數(cdf)
  cdf_min = cdf_max = img.size
  cdf = np.zeros(256)
  cdf[0] = hist[0]
  for i in range(256):
    cdf[i] = cdf[i-1]+hist[i]
    if cdf[i]!=0 and cdf[i]<cdf_min:
      cdf_min = cdf[i]
  h = (cdf-cdf_min)/(cdf_max-cdf_min)
  #將h的結果，乘以(256(灰度範圍的最大值)-1),再四捨五入，得出均衡化值(新的灰度級)
  equ_value = np.around(h*255).astype('uint8')
  result = equ_value[img]
  return result

image = cv2.imread("test.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

res = HistEqua(image_gray)
cv2.imshow("image",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
