from flask import Flask, jsonify, abort, make_response, request, url_for, render_template, send_file
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
    try:
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
                image_file = placeSudokuDigits(img_PT,img)

                """converting the image file to string format To Decode: https://stackoverflow.com/questions/58494586/how-to-return-image-as-part-of-response-from-a-get-request-in-flask
                with open("result.png", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read())"""    
    except KeyError:
        abort(404)
    print(image_file)
    return send_file("result.png")

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error':'Upload the right file format'}), 404)
 

if __name__ == '__main__':
    app.run(debug=True)