import cv2
import sys
import os
from random import randint      


CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)

def detect_faces(image_path,save_img):

	image=cv2.imread(image_path)
	image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
	for x,y,w,h in faces:
	    sub_img=image[y-10:y+h+10,x-10:x+w+10]
	    os.chdir("Faces")
	    cv2.imwrite(str(randint(0,10000))+".jpg",sub_img)
	    os.chdir("../")
	    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
	cv2.imwrite(os.path.join('Tags/'+save_img+'.jpg'), image)
	cv2.imshow("Faces Found",image)
	if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
		cv2.destroyAllWindows()

if __name__ == "__main__":
	
	if not "Faces" in os.listdir("."):
		os.mkdir("Faces")
    
	if len(sys.argv) < 3:
		print("Image Path to be given as System Argument")
		print("\nUsage: python Detect_face.py 'image path' 'save image name' ")
		sys.exit()

	detect_faces(sys.argv[1],sys.argv[2])
