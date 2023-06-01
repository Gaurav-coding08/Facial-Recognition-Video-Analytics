
Experimented on dlib and combination of face_recognition & dlib and custom logic to check whether for any input image the face in it image matches the faces in the video at each time stamp. 

Observation: Main_facial-recognition.py which is combination of face_recognition & dlib library performed better than traditional dlib with high accuracy & less execution time. For instance: Algorithm can process 30 minutes video of 480p in 120 seconds.  

Installations and Virtual Environment-
 Create new virtual environment -
 Virtualenv env 
 source env/bin/activate 
 pip3 install cmake
 pip3 install dlib
 pip3 install numpy
 pip3 install opencv-python
 pip3 install scikit-learn
 pip3 install face_recognition
 
 # Ensure that all these libraries are installed in the lib of your virtual virtualenv, if not then manually install and paste the folders in the env/lib.

Changes in Code
X value is 30. Can change to any value. More X value will result in less execution time and vice versa

Run the main_facial_recognition.py file which is in Facial Recognition folder.(after installation)
 Input the Image path ( Input such an image which has 1 face for the best results)
 Input the Video path  ( For fast results input an 480p video)

Output 
 1. Program will create a txt file and face_check.jpg file and delete it automatically after the execution.
   Kept the above functionality to check the results at each timestamp, If the reader wants to see the results then at the last part of the function remove the os.remove() functions. Don't delete any file manually otherwise os.remove will throw an error. 

 2. Program will return the json file (named as current_date:current_time.json) as the output and all the files which are mentioned in the above point will be removed automatically

 3. Program will also print the JSON file.

Follow the same steps for dlib_Facial_recognition.py file - only difference is that it wont remove 
The txt file automatically.
