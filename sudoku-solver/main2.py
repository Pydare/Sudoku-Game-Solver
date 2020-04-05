import cv2
from backtracking import solve
from main import getCellPositions, extractSudokuDigits
from base import img
from utils import img_PT



#looking for the empty cells and returning
def detectEmptyCell(cell,img):
    pos = []
    img = cv2.resize(img,(252,252))
    img = img[cell[2]+2:cell[3]-3,cell[0]+2:cell[1]-3]
    contours,_ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:

        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            if (w < 15 and x > 2) and (h < 25 and y > 2):#multiplied each number by 9 due to the resized image
                #pos = (x,y,x+w,y+h)
                pos.append((x,y,x+w,y+h))
                break
    if pos == []:    
        return pos
    else:
        return 0

def placeSudokuDigits(img_PT):
    global img
    #we start looking at the middle of the cell as this is where the sudoku digit should be at
    img_PT = cv2.resize(img_PT,(252,252)) #had to reshape the image size to fit the model shape
    img_color = cv2.resize(img,(252,252))
    cells = getCellPositions(img_PT)
    n = 9
    cr = [cells[i:i+n] for i in range(0, len(cells), n)] #cr meaning cells reshaped
    digits = extractSudokuDigits(img_PT)
    solve(digits) #have to look for a way not use this inside this function
    for i in range(len(cr)):
        for j in range(len(cr[i])):
            pos = detectEmptyCell(cr[i][j],img_PT)
            digit_text = digits[i][j]
            if pos == []:
                cv2.putText(img_color, str(digit_text), ((cr[i][j][0]+8),(cr[i][j][2]+19)),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                continue  
    cv2.imshow('puzzle',img_color)
    cv2.waitKey()

