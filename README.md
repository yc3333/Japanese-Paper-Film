## README：

+ #### Set up app environment (python3):

  ```
   pip3 install PySimpleGUI
   pip3 install opencv-python
   pip3 install numpy
   pip3 install matplotlib
   pip3 install scikit-image
   pip3 install python-math
   pip3 install python-vlc
   pip3 install ctypes
  ```

+ #### Compile and Run in terminal:

  ```
   python3 app.py
  ```

+ #### Main Functions Description：

  + ##### score

    Score Fuction gives a score of the current frame's edge based on the standard red line. The more overlaping line area, the higher score. The score is set to be positive (>0, good frames usually have score between 1300-2500)
    The function could be later modified to adjust the sensitivity to the dash line/ solid line by changing the min_line_length and max_line_gap

  + ##### detectBlank
    detectBlank Fuction returns true if detect the perforation, returns flase otherwise

  + ##### detect_red
    detect_red Fuction returns a list of coordinates of the four red lines in the video

+ #### APP Interface Usage：
  + ##### FILE
    input the 1080p version of the file
  + ##### FILE in 4k
    input the higher resolution version of the file
  + ##### Frame Frequency
    input the frequency of the good frames appears in the film
    the input value normally is within 40-60 (smaller value might contain more positive bad frames, larger value might miss some of the good frames)
  + ##### Starting frame number
    input the frequency of picking good frames. The value should be within 40-60 depended on the machine rate
  + ##### Perforation color
    select the perforation color green or blue
  + ##### Adjust Sensitivity of Detecting Perforation Area
    change the sensitiveity of the perforation, larger level means more tolerence to the larger area of perforation color
  + ##### Download folder
    select the destination folder

   




