# Sudoku-Game-Solver
## Introduction
This is a Computer Vision Application that solves a 9x9 Sudoku Puzzle.This application can solve the 3 major difficulties of the puzzle, which include Easy, Medium & Hard. The puzzle is solved by using a Deep Learning Neural Network model to predict the digits in the image. The digits are extracted and solved using Backtracking Algorithm, which is a popular method of solving a sudoku puzzle. The newly found digits are then placed in the empty cells of the puzzle. An extra feature included in this application is the ability to solve the puzzle by reading in live video of the puzzle through the Computer's webcam. Enjoy!

![Image before being solved](https://github.com/Pydare/Sudoku-Game-Solver/blob/master/easy.png)       ![Image after being solved](https://github.com/Pydare/Sudoku-Game-Solver/blob/master/solved_easy.png)
## Dependency
- Python3, Ubuntu 18.04 or WindowsOS
- OpenCV, Tensorflow, Keras, Pillow
- To install the required packages, run pip install -r requirements.txt
## Dataset for Training Model
The CNN model used for this application is the Keras MNIST Handwritten Digit Classification. Another option could be to use the
Chars 74K Dataset, which contains characters from computer fonts with 4 variations (combinations of italic, bold and normal).
The download link: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
## Usage (Upload)
- First, clone the repository with git clone https://github.com/Pydare/Sudoku-Game-Solver.git and enter the cloned folder.
- Create a virtual environment containing the requirement libraries from the requirements.txt file and activate it.
- Upload a sudoku image you want to solve. Some sample images are in the repo, the 3 major difficulties (easy,medium & hard)
- Open the sudoku-solver directory, in the base.py file, enter the name of the image file in the second line.
- Solve the puzzle using python run.py in your terminal/bash
## Usage (Using Webcam)
- First, clone the repository with git clone https://github.com/Pydare/Sudoku-Game-Solver.git and enter the cloned folder.
- Create a virtual environment containing the requirement libraries from the requirements.txt file and activate it.
- Facing your webcam with an image of the puzzle, enter python sudokuWebcam.py in your terminal and the puzzle is solved

The jupyter notebook file can also be used for step-by-step understanding of how the application was built and solve the puzzle via both methods.

