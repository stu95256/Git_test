import numpy as np
import cv2

#腐蝕 Erode
def erode(img,kernel,Erode_time=1):
  H, W = img.shape
  kh, kw = kernel.shape
  out = img.copy()
  #圖片邊緣填充((上,下),(左,右))
  pad_width = ((int(kh/2),int(kh/2)),(int(kw/2),int(kw/2)))
  #跌代次數
  for i in range(Erode_time):
    #圖片邊緣填充
    tmp = np.pad(out,pad_width,'constant',constant_values=255)
    #原圖片每一格
    for x in range(H):
      for y in range(W):
        #每一格計算
        if np.mean(kernel*tmp[x:x+kh, y:y+kw]) < 255:
          out[x, y] = 0
  return out

#膨脹 Dilate
def dilate(img,kernel,Dilate_time=1):
  H, W = img.shape
  kh, kw = kernel.shape
  out = img.copy()
  #圖片邊緣填充((上,下),(左,右))
  pad_width = ((int(kh/2),int(kh/2)),(int(kw/2),int(kw/2)))
  #跌代次數
  for i in range(Dilate_time):
    #圖片邊緣填充
    tmp = np.pad(out,pad_width,'constant',constant_values=0)
    #原圖片每一格
    for y in range(H):
      for x in range(W):
        #每一格計算
        if np.sum(kernel*tmp[y:y+kh, x:x+kw]) >= 255:
          out[y, x] = 255
  return out

image = cv2.imread("test.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_bin = cv2.threshold(image_gray, 50, 255, cv2.THRESH_OTSU)[1]

kernel = np.ones((3,3),np.uint8)
res_erode = erode(image_bin,kernel,1)
res_dilate = dilate(image_bin,kernel,1)
cv2.imshow("image_erode",res_erode)
cv2.imshow("image_dilate",res_dilate)
cv2.waitKey(0)
cv2.destroyAllWindows()
