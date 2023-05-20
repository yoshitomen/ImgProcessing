# -*- coding: utf-8 -*-
import cv2, os

path =os.path.dirname(os.path.abspath(__file__))+"/"
sub_dir="report/"

image_file = []
image_file.append("sample.jpg")

image_path = []
for idt in range(len(image_file)):
   image_path.append(path + sub_dir + image_file[idt])
   print("読み込んだファイル："+image_file[idt])
out_image_path = []
for idt in range(len(image_file)):
   out_image_path.append(path + sub_dir + image_file[idt])

image = []
for idt in range(len(image_file)):
   image.append(cv2.imread(image_path[idt])) # 画像ファイル読み込み
#print('JPG data : '+image)
#cv2.imwrite(out_image_path, image)

gray = []
for idt in range(len(image_file)):
   gray.append(cv2.cvtColor(image[idt], cv2.COLOR_RGB2GRAY)) # グレースケースに変換
   cv2.imwrite(path+sub_dir+"only_gray.jpg", gray[0])
   gray[idt]=cv2.GaussianBlur(gray[idt],(3,3),3) #gaussianフィルタで平滑化(ノイズ除去)
laplacian = []
print("laplacian.var()の値")
for idt in range(len(image_file)):
   laplacian.append(cv2.Laplacian(gray[idt], cv2.CV_64F)) #ラプラシアンフィルタを適用
   print(image_file[idt]+" = ",laplacian[idt].var())
   #各画素のラプラシアンを集計して分散を出力(ラプラシアン値).これが大きい合焦していることになる.
"""
デフォルトのLaplacianは4近傍3*3カーネルの畳み込み。8近傍で計算したかったらカーネルを用意する必要あり
kernel = np.array([[1, 1,  1],
                   [1, -8, 1],
                   [1, 1,  1]])

"""

"""
if laplacian.var() < 100: # 閾値は100以下だとピンボケ画像と判定
   text = "Blurry"
else:
   text = "Not Blurry"

cv2.putText(image, "{}: {:.2f}".format(text, laplacian.var()), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
"""

cv2.imwrite(out_image_path[0], gray[0])
cv2.imwrite(path+sub_dir+"sample_edge.jpeg", 255-laplacian[0])
#ラプラシアンフィルタ後の画像を反転して保存
