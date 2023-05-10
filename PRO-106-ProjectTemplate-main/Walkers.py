import cv2


# Create our body classifier
body_classifier =cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
   
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    
    # Extract bounding boxes for any bodies identified
    def drawBox(img, bbox):
      x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

      cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)

    cv2.imshow("result",frame)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
