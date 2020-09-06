from flask import Flask, jsonify, abort, make_response, request, url_for, render_template
import cv2
import base64
import numpy as np
from sudoku_pkg.utils import preprocessImage, probHoughTransformUtil, four_point_transform
from sudoku_pkg.main import prediction, getCellPositions, predictDigit, extractSudokuDigits
from sudoku_pkg.backtracking import solve
from sudoku_pkg.main2 import detectEmptyCell, placeSudokuDigits

app = Flask(__name__) 

@app.route('/upload-image', methods=['GET','POST'])
def upload_image():
    
    if request.method == "POST":
        if request.files:
            #  read encoded image
            image_string = request.files["image"].read()

            #  convert binary data to numpy array
            nparr = np.fromstring(image_string, np.uint8)

            #  let opencv decode image to correct format
            img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

            #preprocessing the image
            thresh_inv = preprocessImage(img)
            poly_approx = probHoughTransformUtil(thresh_inv,img)
            #perspective transformed image
            img_PT = four_point_transform(thresh_inv,poly_approx)

            #sudoku solved
            placeSudokuDigits(img_PT,img)


    return("There is an image")


if __name__ == '__main__':
    app.run(debug=True)