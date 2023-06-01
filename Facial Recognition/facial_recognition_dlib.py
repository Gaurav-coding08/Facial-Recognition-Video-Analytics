# for faces in image.
import dlib
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import time
import json


img = input("PASTE IMAGE PATH")
vid= input("PASTE VIDEO PATH")


def returnJson(img,vid):
    start = time.time()

    now = datetime.datetime.now()
    file_name = f"{now.date()}_{now.hour}:{now.minute}:{now.second}.txt"

    #change paths accordingly
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("/Users/gauravrastogi/Desktop/dlib_Facial Recognition/dlib_facial_recog/shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("/Users/gauravrastogi/Desktop/dlib_Facial Recognition/dlib_facial_recog/models/dlib/dlib_face_recognition_resnet_model_v1.dat")
    
    
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    dets = detector(image)

    # Loop over detected faces
    # it is compulsory that image you import has face.
    for det in dets:
        # Get face landmarks
        shape = sp(gray, det)
         
        descriptor = facerec.compute_face_descriptor(image, shape)

    print(descriptor) 

    #convert descriptor into array    
    face_descriptor_1 = np.array(descriptor)     
    
        
    # for faces in video
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/gauravrastogi/Desktop/dlib_Facial Recognition/dlib_facial_recog/shape_predictor_68_face_landmarks.dat")
    facerec =dlib.face_recognition_model_v1("/Users/gauravrastogi/Desktop/dlib_Facial Recognition/dlib_facial_recog/models/dlib/dlib_face_recognition_resnet_model_v1.dat")
    #detector = dlib.cnn_face_detection_model_v1("/Users/gauravrastogi/Desktop/Facial Recoognition/facial_recog/models/dlib/mmod_human_face_detector.dat")


    # Create a video capture object and set the resolution
    cap = cv2.VideoCapture(vid)

    X =30

    double_check=[]
    timess=[]
    triple_check=[]
    timesstriple=[]

    # # Loop through the frames of the video

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
        
            
            with open(file_name, "a") as file:
                file.write("\n")
                
            with open(file_name, "a") as file:  
                file.write("FACE MATCHING RESULTS ARE ABOVE" + "\n")
                
            with open(file_name, "a") as file:
                file.write("\n")
                
            with open(file_name, "a") as file:  
                file.write("ADDITIONAL RESULTS ARE BELOW" + "\n")  
                
            
            break
        
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % X != 0:
            continue
        
        if ret:
            #If the frame was successfully read, convert it to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Detect faces in the frame
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            faces = detector(gray)
            #trial
            faces = detector(frame)
            if len(faces) > 1:
                
                double_check.append(frame)
                timess.append(timestamp_sec)
            # Loop through the detected faces
            
            for face in faces:
            
               
                shape = predictor(gray, face)
                    
                descriptor2 = facerec.compute_face_descriptor(gray, shape)
                face_descriptor_2 = np.array(descriptor2)
                    
                similarity = cosine_similarity(face_descriptor_1.reshape(1, -1), face_descriptor_2.reshape(1, -1))
                distance = 1 - similarity[0][0]
                print("distance 1:",distance)

                if distance < 0.12:
                    with open(file_name, "a") as file:
                #  print("Multiple faces detected!",timestamp_sec)
                     file.write("Face Match:" + str(timestamp_sec) + "\n")
                    
                else:
                
                            double_check.append(frame)
                            timess.append(timestamp_sec)

            # Exit if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()   
    i=0

    if not double_check: 
        # with open(file_name, "a") as file:
        #     file.write("\n")

        with open(file_name, "a") as file:
            file.write("\n")
            
        with open(file_name, "a") as file:
            file.write("PASSED")
            
    else:    

        with open(file_name, "a") as file:
            file.write("\n")
            
        with open(file_name, "a") as file:      
            file.write("DOUBLE CHECK VERIFICATION RESULTS-" + "\n")
                        
        for frame in double_check:
                faces = detector(frame)  
                if faces: 
                    if len(faces) > 1:
                            triple_check.append(frame)
                            timesstriple.append(timess[i])
                            
                            with open(file_name, "a") as file:
                    #print("Multiple faces detected!",timestamp_sec)
                             file.write("Multiple Faces Detected:" + str(timess[i]) + "\n")
                            # plt.imshow(frame, cmap='gray')
                            # plt.show()  
                    shape = predictor(frame, face)
                    descriptor2 = facerec.compute_face_descriptor(frame, shape)
                    face_descriptor_2 = np.array(descriptor2)
                                
                    similarity = cosine_similarity(face_descriptor_1.reshape(1, -1), face_descriptor_2.reshape(1, -1))
                    distance = 1 - similarity[0][0]
                    distance2=np.sum(np.abs(face_descriptor_1 - face_descriptor_2))
                    print("distance 2:" , distance) 
                    # earlier 0.1
                    if distance < 0.1:
                                    # Draw a green rectangle around the face if it matches
                                   
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        with open(file_name, "a") as file:
                    
                            file.write("Face Match:" + str(timess[i]) + "\n")
                                    
                    else:
                        
                    #   print("Face do not match! at :",timess[i])
                            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                            cv2.imshow('video',frame)
                            #   plt.imshow(frame, cmap='gray')
                            #   plt.show() 
                            triple_check.append(frame)
                            timesstriple.append(timess[i])
                            with open(file_name, "a") as file:
                            
                                file.write("Face Do Not Match:" + str(timess[i]) + "\n")
                            
                    
                else:
                    continue
                i=i+1
    z=0         

    
    if not triple_check:  

            
     with open(file_name, "a") as file:
            file.write("\n")

     with open(file_name, "a") as file:
            file.write("TRIPLE CHECK VERIFICATION RESULTS --" + "\n")
            
     with open(file_name, "a") as file:
            file.write("PASSED" +  "\n")

    else:    
     with open(file_name, "a") as file:
            file.write("\n")

     with open(file_name, "a") as file:
            file.write("TRIPLE CHECK VERIFICATION RESULTS --" + "\n")



            
     for framee in triple_check:
                faces = detector(framee)  
                if faces: 
                    if len(faces) > 1:
                            
                            with open(file_name, "a") as file:
                             file.write("Multiple Faces Detected:" + str(timesstriple[z]) + "\n") 
                    shape = predictor(framee, face)
                    descriptor2 = facerec.compute_face_descriptor(framee, shape)
                    face_descriptor_2 = np.array(descriptor2)
                                        
                    similarity = cosine_similarity(face_descriptor_1.reshape(1, -1), face_descriptor_2.reshape(1, -1))
                    distance = 1 - similarity[0][0]
                    distance2=np.sum(np.abs(face_descriptor_1 - face_descriptor_2))
                    print("distance 3" , distance) 
                           
                    if distance < 0.12:
                                    
                        cv2.rectangle(framee, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        with open(file_name, "a") as file:
                    #print("Multiple faces detected!",timestamp_sec)
                            file.write("Face Match:" + str(timesstriple[z]) + "\n")
                                    
                    else:

                            cv2.rectangle(framee, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                            cv2.imshow('video',framee)

                            with open(file_name, "a") as file:
                                file.write("Face Do Not Match:" + str(timesstriple[z]) + "\n")

                else:
                        # print("no faces")
                         continue
                z=z+1
            

    # search_text = "TRIPLE CHECK VERIFICATION RESULTS --" 
    
    
    # returning JSON
    output_filename = f"{now.date()}_{now.hour}:{now.minute}:{now.second}.json"
    dot_index = output_filename.index(".")
    videoid = output_filename[:dot_index]
    
    x = None
    y = None
    z = None
    Result=None
    face_do_not_match_times=[]
    multiple_face_detected_times=[]
    i1=0
    i2=0
    tempmulti=0
    tempmismatch=0
    i=0
    
    with open(file_name, "r") as f:
            lines = [line.strip() for line in f.readlines()]
    
    for i in range(len(lines)):
                for line in lines[i+1:]:
                    if line.startswith("Multiple Faces Detected"):
                            z = float(line.split(":")[-1].strip()) 
                            multiple_face_detected_times.append(z)   
                              
    with open(file_name, "r") as f:
            lines = [line.strip() for line in f.readlines()]
    
    for i in range(len(lines)):
                                                      
                if lines[i].startswith("TRIPLE CHECK VERIFICATION RESULTS --"):
                    print("found")
                    for line in lines[i+1:]:
                        if line.startswith("Face Do Not Match") :
                          x = float(line.split(":")[-1].strip())
                          face_do_not_match_times.append(x)
                        if line.startswith("Face Match"):
                          y = float(line.split(":")[-1].strip())
                        if line.startswith("Multiple Faces Detected"):
                          z = float(line.split(":")[-1].strip())
                          multiple_face_detected_times.append(z)
                        elif "PASSED" in line:
                            
                          Result = "PASSED"
                          x=None
                          z=None
                          y="All"
                          break
    
    with open(file_name, "r") as f:
            lines = [line2.strip() for line2 in f.readlines()]
    i=0        
    for i in range(len(lines)):
                if lines[i].startswith("TRIPLE CHECK VERIFICATION RESULTS --"):
                    templine=lines[i+2:]
                    print(templine)
                    if all(line.startswith("Face Match") for line in templine):
                        if len(multiple_face_detected_times) == 0:
                          Result = "PASSED"                     
                    break
                
    unique_facedonotmatchtimes = set(face_do_not_match_times)        
    face_do_not_match_times = np.array(list(unique_facedonotmatchtimes))    
     
    unique_facemultipleface = set(multiple_face_detected_times)        
    multiple_face_detected_times = np.array(list(unique_facemultipleface))
    
            
    if Result == "PASSED":       
        output = {
    "video_id": videoid,
    "face_match": "true",
    "multi_face_detected": "false",
     "multi_face_detailed_report": [],
    "face_mismatch_detailed_report": []}
   
        with open(output_filename, "w") as f:
          json.dump(output, f,indent=4)
    
    elif len(multiple_face_detected_times) == 0 and len(face_do_not_match_times) == 0:
        output = {
    "video_id": videoid,
    "face_match": "true",
    "multi_face_detected": "false",
     "multi_face_detailed_report": [],
    "face_mismatch_detailed_report": []}
   
        with open(output_filename, "w") as f:
          json.dump(output, f,indent=4)
          
          
    
    elif len(multiple_face_detected_times) == 0:
        output = {
    "video_id": videoid,
    "face_match": "false",
    "multi_face_detected": "false",
    "multi_face_detailed_report": [],
    "face_mismatch_detailed_report": [] }
        
        while i1 < len(face_do_not_match_times):
            tempmismatch=face_do_not_match_times[i1]
            output_mismatch_detailed ={
                     "start_timestamp": tempmismatch,
                    "face_mismatch_detected": "true"
                      }
            output["face_mismatch_detailed_report"].append(output_mismatch_detailed)
            i1 += 1
        
        with open(output_filename, "w") as f:
          json.dump(output, f,indent=4)
          
    else:
       output = {
    "video_id": videoid,
    "face_match": "false",
    "multi_face_detected": "true",
    "multi_face_detailed_report": [],
    "face_mismatch_detailed_report": [] }
       
       while i1 < len(face_do_not_match_times):
            tempmismatch=face_do_not_match_times[i1]
            output_mismatch_detailed ={
                     "start_timestamp": tempmismatch,
                    "face_mismatch_detected": "true"
                      }
            output["face_mismatch_detailed_report"].append(output_mismatch_detailed)
            i1 += 1

       while i2 < len(multiple_face_detected_times):
            tempmulti=multiple_face_detected_times[i2]
            output_multi_detailed ={
                 "start_timestamp": tempmulti,
                 "multi_face_detected": "true"
                  }
            output["multi_face_detailed_report"].append(output_multi_detailed)
            i2 += 1
       with open(output_filename, "w") as f:
          json.dump(output, f,indent=4)
            
    
   
     
    end = time.time()
    print("Execution Time:",end-start)
    
    print(output)
    print("JSON file created successfully!")


returnJson(img,vid)
