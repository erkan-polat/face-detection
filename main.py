import cv2
from icrawler.builtin import GoogleImageCrawler
import os
import numpy as np

"""
google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': 'pos'})
filters = dict(
    size='large',
    date=(None, None)
)

google_crawler.crawl(keyword='human face', filters=filters,  max_num=1000, file_idx_offset=0)
"""


"""
list = []
for x in os.listdir('pos'):
    if x.endswith(".jpg"):
        list.append(x)
print(list[0])
print(type(list[0]))

for i in range(len(list)):
    path_p = "Positive//"
    path_p1= (os.path.join(path_p,list[i]))
    image = cv2.imread(path_p1)
    path_n = "neg//"
    path_n1= (os.path.join(path_n,list[i]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path_n1,gray)

"""

"""
google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=2,
    downloader_threads=4,
    storage={'root_dir': 'neg'})
filters = dict(
    size='large',
    date=(None, None)
)

google_crawler.crawl(keyword='cat face', filters=filters,  max_num=1000, file_idx_offset=0)
"""
"""
def create_pos():
    for img in os.listdir("neg"):
        line = "neg"+'/'+img+'\n'
        with open('bg.txt','a') as f:
            f.write(line)
    for img in os.listdir("pos"):
        line = "pos"+'/'+img+' 1 0 0 50 50\n'
        with open('info.txt','a') as f:
            f.write(line)
create_pos()

"""

""" opencv_createsamples -img Erkan.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950
"""

"""opencv_createsamples -info info/info.lst -num 1950 -w 20 -h 20 -vec positives.vec


"""

"""opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 60 -numNeg 60  -numStages 10 -w 20 -h 20
"""


face_cascade = cv2.CascadeClassifier('opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')
Erkan_cascade = cv2.CascadeClassifier('opencv/build/etc/haarcascades/Erkan.xml')

img = cv2.imread('Erkan.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1, 4)
Erkan = Erkan_cascade.detectMultiScale(gray,1.3,20)

for (x, y, w, h) in Erkan:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.putText(img, "Erkan", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    cv2.putText(img, "Face", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()