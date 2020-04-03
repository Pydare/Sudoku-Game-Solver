from main import getCellPositions, extractSudokuDigits
from main2 import detectEmptyCell
from backtracking import solve
from utils import four_point_transform
import cv2
import numpy as np
from tensorflow.keras.models import load_model




def placeSudokuDigitsLive(img_PT):
    #we start looking at the middle of the cell as this is where the sudoku digit should be at
    img_PT = cv2.resize(img_PT,(252,252)) #had to reshape the image size to fit the model shape
    img_color = cv2.resize(frame,(252,252)) #img is got from the grayscale
    cells = getCellPositions(img_PT)
    n = 9
    cr = [cells[i:i+n] for i in range(0, len(cells), n)] #cr meaning cells reshaped
    digits = extractSudokuDigits(img_PT)
    solve(digits) 
    for i in range(len(cr)):
        for j in range(len(cr[i])):
            pos = detectEmptyCell(cr[i][j],img_PT)
            digit_text = digits[i][j]
            if pos == []:
                cv2.putText(img_color, str(digit_text), ((cr[i][j][0]+8),(cr[i][j][2]+19)),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                continue  
    


## PUT THIS ALL IN ONE CELL!

import cv2

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#to run some lines once
flag = True

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #Convert the captured frame into grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    #This segment of the code works on the board segment of the frame
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    # find the biggest area
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    poly_approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    board_segment = four_point_transform(gray,poly_approx)

    #Applying Gaussian Blurring to the image
    dst = cv2.GaussianBlur(board_segment,(1,1),cv2.BORDER_DEFAULT)
    
    #Applying Inverse Binary Threshold to the image
    ret,thresh_inv = cv2.threshold(dst, 180, 255,cv2.THRESH_BINARY_INV)
    
    #Applying Probabilistic Hough Transform on the Binary Image
    minLineLength = 100
    maxLineGap = 60
    lines = cv2.HoughLinesP(thresh_inv,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for l in lines:
        x1,y1,x2,y2 = l[0]
        cv2.line(board_segment,(x1,y1),(x2,y2),(0,255,0),2, cv2.LINE_AA)
        
    if flag:
        #using neural network model to detect the digits in the image
        #new_model = load_model('keras_digit_model.h5')
        a = extractSudokuDigits(thresh_inv)

        #solving with backtracking
        solve(a)

        #putting back the solved digits on spaces that are empty
        placeSudokuDigitsLive(thresh_inv) #this function won't have the plt.imshow() and also, 
                                    #the colored image would be img
        
        flag = False
        
    #overlaying the board segment of the image on the frame
    x_offset, y_offset = (poly_approx[0][0].tolist()[0]),(poly_approx[0][0].tolist()[1])
    x_end, y_end = (x_offset+board_segment.shape[1]), (y_offset+board_segment.shape[0])
    frame[y_offset:y_end,x_offset:x_end] = board_segment
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()