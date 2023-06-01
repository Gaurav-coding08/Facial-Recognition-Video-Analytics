import dlib
import cv2
import datetime
import time
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.spatial.distance import cosine
import json


now = datetime.datetime.now()
file_name = f"{now.date()}_{now.hour}:{now.minute}:{now.second}.txt"


img = input("PASTE IMAGE PATH")
vid= input("PASTE VIDEO PATH")

def DetectCheating(img,video):
    start = time.time()
    face_detector = dlib.get_frontal_face_detector()
    image = cv2.imread(img)
    faces = face_detector(image)
    
    # print("no of faces in the Image",len(faces))
    
    for face in faces:
        
        temp_face_image_check = image[face.top():face.bottom(), face.left():face.right()]
        
        cv2.imwrite("face_check.jpg", temp_face_image_check)
       
    temp_face_image_check = cv2.cvtColor(temp_face_image_check, cv2.COLOR_BGR2RGB)
    known_face_encoding = face_recognition.face_encodings(temp_face_image_check)[0]


    video = cv2.VideoCapture(vid)

    #speed
    X=30

    
    times_frames_with_mfd=[]
    times_frames_with_fmm=[]
    count=0
    print("Evaluating the results...")
    while video.isOpened():
        
        ret, frame = video.read()
        if not ret:
                with open(file_name, "a") as file:
                    file.write("\n")
                    
                with open(file_name, "a") as file:  
                    file.write("RESULTS ARE ABOVE" + "\n")
                    
                with open(file_name, "a") as file:
                    file.write("\n")
                    
                break
            
        if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % X != 0:
                continue

        if ret:

    #trial
                timestamp_sec = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                faces = face_detector(frame)
                
                if len(faces) == 0:
                    # print("No face")
                    continue            
                else:
    #trial      
                 try:
                    if len(faces)>1:
               
                            
                            with open(file_name, "a") as file:
                                file.write("multiple Face Detected:" + str(timestamp_sec) + "\n")
                            times_frames_with_mfd.append(timestamp_sec)
                            count+=1 
                    for face in faces:   
                            
                        temp_face_image_video = frame[face.top():face.bottom(), face.left():face.right()]
                        temp_face_image_video = cv2.cvtColor(temp_face_image_video, cv2.COLOR_BGR2RGB)
                        try:
                                current_face_encoding = face_recognition.face_encodings(temp_face_image_video)[0]
                                results = face_recognition.compare_faces([known_face_encoding], current_face_encoding)
                                # print(results)
                                distance_cosinedistance = cosine(known_face_encoding, current_face_encoding)
                                
                                if results[0]:
                                    with open(file_name, "a") as file:
                                        file.write("Face match:" + str(timestamp_sec) + "\n")
                                    count+=1
                                else:       
                                    if distance_cosinedistance <= 0.11369517114668837:
                                        with open(file_name, "a") as file:
                                            file.write("Face match:" + str(timestamp_sec) + "\n")   
                                        count+=1  
                                    else:
                                        with open(file_name, "a") as file:
                                            file.write("Face mismatch" + str(timestamp_sec) + "\n")
                                        times_frames_with_fmm.append(timestamp_sec)
                                        count+=1
                        except:
                              continue
                 except:
                    continue
    # Close the video
    video.release()
    cv2.destroyAllWindows()


    # returning & making JSON
    output_filename = f"{now.date()}_{now.hour}:{now.minute}:{now.second}.json"
    dot_index = output_filename.index(".")
    videoid = output_filename[:dot_index]

    #logic if both are not there
    if len(times_frames_with_mfd)==0 and len(times_frames_with_fmm) ==0:
        output = {
        "video_id": videoid,
        "face_match": "true",
        "multi_face_detected": "false",
        "multi_face_detailed_report": [],
        "face_mismatch_detailed_report": []}
        
        with open(output_filename, "w") as f:
            json.dump(output, f,indent=4)
            
    #logic if multiple faces are not there , but face mismatch , return json accordingly
    elif len(times_frames_with_mfd)==0 and len(times_frames_with_fmm) !=0:
        
        check=count//8
        if len(times_frames_with_fmm) <= check:
            output = {
        "video_id": videoid,
        "face_match": "true",
        "multi_face_detected": "false",
        "multi_face_detailed_report": [],
        "face_mismatch_detailed_report": []}
    
            with open(output_filename, "w") as f:
             json.dump(output, f,indent=4)
        
        else:
            output = {
        "video_id": videoid,
        "face_match": "false",
        "multi_face_detected": "false",
        "multi_face_detailed_report": [],
        "face_mismatch_detailed_report": []}
            
            for value in times_frames_with_fmm:
                tempmismatch=value
                output_mismatch_detailed ={
                        "start_timestamp": tempmismatch,
                        "face_mismatch_detected": "true"
                        }
                output["face_mismatch_detailed_report"].append(output_mismatch_detailed)
            
            with open(output_filename, "w") as f:
             json.dump(output, f,indent=4)    
            
    #logic if multple faces are there but mismatch is not there
    elif len(times_frames_with_mfd) !=0 and len(times_frames_with_fmm) ==0:
        
        output = {
        "video_id": videoid,
        "face_match": "false",
        "multi_face_detected": "true",
        "multi_face_detailed_report": [],
        "face_mismatch_detailed_report": []}
        
        for value_mfd in times_frames_with_mfd:
                tempmismfd=value_mfd
                output_mfd_detailed ={
                        "start_timestamp": tempmismfd,
                        "multi_face_detected": "true"
                        }
                output["multi_face_detailed_report"].append(output_mfd_detailed)
        
        with open(output_filename, "w") as f:
            json.dump(output, f,indent=4)

    else:
        
        output = {
        "video_id": videoid,
        "face_match": "false",
        "multi_face_detected": "true",
        "multi_face_detailed_report": [],
        "face_mismatch_detailed_report": []}
        
        for value_fmm in times_frames_with_fmm:
                tempfmm=value_fmm
                output_mismatch_detailed ={
                        "start_timestamp": tempfmm,
                        "face_mismatch_detected": "true"
                        }
                output["face_mismatch_detailed_report"].append(output_mismatch_detailed)
        
        for value_mfd in times_frames_with_mfd:
                tempmismfd=value_mfd
                output_mfd_detailed ={
                        "start_timestamp": tempmismfd,
                        "multi_face_detected": "true"
                        }
                output["multi_face_detailed_report"].append(output_mfd_detailed)
        
        with open(output_filename, "w") as f:
            json.dump(output, f,indent=4)

    print(output)
    print("JSON file created successfully!")
    
    end = time.time()
    print("Execution Time:",end-start)
    
DetectCheating(img,vid)      
      

