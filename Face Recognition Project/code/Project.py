# -*- coding: utf-8 -*-
"""
@author: JohnP
"""

#importing the needed libraries
import cv2
import face_recognition
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition

#Initialising the video from external camera
webcam_video_stream = cv2.VideoCapture(0)
me_image = face_recognition.load_image_file(r'C:\Users\JohnP\Face\tech\Face Recognition Project\code\me1.jpg')#Initialising me_image to save my picture as a variable
#In the code above I have attached an "r" to only allow the file to be read an not need a response
me_image_encoding = face_recognition.face_encodings(me_image)[0]
#The code above is loading the images of myself 

#The code below is saving the encodings and labels in a single array
face_encodings = [me_image_encoding]
face_names = ["John Phillips"]

#Holding all face locations, encodings and names in the square frame
all_locations = []
all_encodings = []
all_usernames = []

#This code will loop through each frame in the video
while True:
    ret,frame = webcam_video_stream.read()#This will get the current frame from 
                                                    #the video as an image
    small_frame = cv2.resize(frame,(0,0),fx=0.30,fy=0.30)#I have adjusted fx/fy to depend on the face to webcam distance
    #The code above will detect all faces in the video
    all_locations = face_recognition.face_locations(small_frame,number_of_times_to_upsample=1,model='hog')
    #number of times to upsample still allows the facial recognition software to identify a face no matter how far it is away
    # from webcam, it will also still attempt to identify an image if e.g. a hand is infront   
    all_encodings = face_recognition.face_encodings(small_frame,all_locations)



    for current_location,current_encoding in zip(all_locations,all_encodings):
    #Identifying top,right,bottom and left variables for each facial position   
        top_pos,right_pos,bottom_pos,left_pos = current_location
        
      
        top_pos = top_pos*3
        right_pos = right_pos*3
        bottom_pos = bottom_pos*3
        left_pos = left_pos*3
        #I have multiplied each pos by 3 so it fits the separate window as with no multiplication
        #it causes the video to only show a portion of the video
        all_matches = face_recognition.compare_faces(face_encodings, current_encoding)
        
        #Blurring code // Uncomment to use below
        #face_image = cv2.GaussianBlur(face_image, (95,95), 30) #Adding blur effect to face_image variable
    
        status = 'Unverified' #This is for if someones face is not in the database
        
      #The code below is checking whether all of the matches have at least one item
        if True in all_matches: #if true then it will get the index number of the face that is in the first index
            first_match_index = all_matches.index(True)
            status = face_names[first_match_index]
        
       #This is drawing the rectangle around the persons face
        cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        font = cv2.FONT_HERSHEY_TRIPLEX #Colour of the font
        #This is displaying the name of the person in the left bottom corner of the rectangle
        cv2.putText(frame, status, (left_pos,bottom_pos), font, 0.6, (255,255,255),1)
    
  #This will display the video from the webcam
    cv2.imshow("Webcam Video",frame)
    #When the user presses 'x' on their keyboard, the video window will close
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


webcam_video_stream.release()
cv2.destroyAllWindows()        
