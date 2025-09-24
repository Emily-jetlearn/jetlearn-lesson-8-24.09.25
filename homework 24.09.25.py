import cv2
import numpy as npy



dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",1)
cv2.imshow("colour dog",dog)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",0)
cv2.imshow("greyscale dog",dog)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",-1)
cv2.imshow("unchanged dog",dog)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",1)
B,G,R=cv2.split(dog)
cv2.imshow("blue saturation dog", B)
cv2.waitKey(0)
cv2.imshow("green saturation dog", G)
cv2.waitKey(0)
cv2.imshow("red saturation dog", R)
cv2.waitKey(0)
cv2.destroyAllWindows()

image1=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",-1)
image2=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\resized forest.jpg",-1)
sumimage=cv2.addWeighted(image1,0.6,image2,0.5,0)
cv2.imshow("image1", image1)
cv2.imshow("image2", image2 )
cv2.imshow("adding images",sumimage)
cv2.waitKey(0)
cv2.destroyAllWindows()


dog_image=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",-1)
forest_image=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\forest image.jpg",-1)

subimage=cv2.subtract(forest_image,dog_image)
cv2.imshow("subtracted", subimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",-1)
resized=cv2.resize(dog,(300,900))
cv2.imshow("resized dog",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png",-1)
erosion=npy.ones((5,5),npy.uint8)
doggy=cv2.erode(dog,erosion)
cv2.imshow("eroded dog",doggy)
cv2.waitKey(0)
cv2.destroyAllWindows()



gaussian=cv2.GaussianBlur(dog,(7,7),0)
cv2.imshow("dog gaussian blur",gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()

median=cv2.medianBlur(dog,5)
cv2.imshow("dog median blur",median)
cv2.waitKey(0)
cv2.destroyAllWindows()

bilateral=cv2.bilateralFilter(dog,9,75,75)
cv2.imshow("dog bilateral filter",bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

border=cv2.copyMakeBorder(dog,7,7,7,7,cv2.BORDER_CONSTANT,value=100)
cv2.imshow("dog bordered",border)
cv2.waitKey(0)
cv2.destroyAllWindows()

reflective_border=cv2.copyMakeBorder(dog,300,300,300,300,cv2.BORDER_REFLECT,value=1)
cv2.imshow("dog reflective border",reflective_border)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png")
(row,column)=dog.shape[:2]
position=cv2.getRotationMatrix2D((column/2,row/2),45,0.5) 
result=cv2.warpAffine(dog,position,(column,row))
cv2.imshow("rotated dog half",result)

position=cv2.getRotationMatrix2D((column/2,row/2),45,1) 
result=cv2.warpAffine(dog,position,(column,row))
cv2.imshow("rotated dog",result)

position=cv2.getRotationMatrix2D((column/2,row/2),45,2) 
result=cv2.warpAffine(dog,position,(column,row))
cv2.imshow("rotated dog double",result)
cv2.waitKey(0)
cv2.destroyAllWindows()

dog=cv2.imread(r"C:\Users\olada\OneDrive\Desktop\Opencv\opencvimages\dogimg.png")
hsvimg=cv2.cvtColor(dog,cv2.COLOR_BGR2HSV)
cv2.imshow("rgb to hsv",hsvimg)
cv2.waitKey(0)
cv2.destroyAllWindows()