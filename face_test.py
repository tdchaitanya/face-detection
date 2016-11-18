#Dependencies : opencv,dlib
import matplotlib.pyplot as plt
import dlib
import urllib
import cv2
import numpy as np
from PIL import Image
import cStringIO

# use the below url  or enter your own
URL_to_test = 'http://images.huffingtonpost.com/2016-07-15-1468607338-43291-DonaldTrumpangry.jpg'
file = cStringIO.StringIO(urllib.urlopen(URL_to_test).read())
img = Image.open(file)
orig_img = img.copy()
img = np.asarray(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
dets = detector(gray,1)
print('Done')

faces = []
for k,d in enumerate(dets):
    x = d.left()
    y = d.top()
    w = d.right() - d.left()
    h = d.top()- d.bottom()
    cv2.rectangle(img,(x,y),(x-h,y-h),(255,0,0),5)

fig = plt.figure(figsize = (15,15))
ax1 = fig.add_subplot(211)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Original Image')
ax1.imshow(orig_img)
#plt.show()

ax2 = fig.add_subplot(212)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Image with Detections')
ax2.imshow(img)
plt.show()
