# Leukemia Cell Classifier

This is a Streamlit app that takes in an image of a cell and identifies whether the cell is cancerous or healthy. <br/>

Kaggle data set used: https://www.kaggle.com/datasets/andrewmvd/leukemia-classification/data

## Files
- **App.py**: Contains the code for the Streamlit app. Running this file through the instructions stated below should open the app on your device.
- **Testing.py**: Used for testing the model. Takes in an image of a cell and identifies whether it is cancerous (prints 0) or healthy (prints 1).
- **Training.py**: Contains the code used to build and train the model.
- **model.pth**: Stores the model created and trained by Training.py and used by Testing.py and App.py.
- **CellClassifier.py**: Contains the class used to define the model in Training.py, App.py, and Testing.py.
- 
## Setting Up
After downloading the directory, do the following: <br/>
1. Install necessary libraries: streamlit, torch, pillow, dotenv <br/>
     &emsp; - To install Streamlit, run the following command in your terminal: **pip install streamlit** <br/>
     &emsp;&emsp;&emsp;To test that Streamlit has been successfully installed, run the following command in your terminal: **streamlit hello** <br/>
     &emsp;&emsp;&emsp;For more help on installing Streamlit, refer to this link: https://docs.streamlit.io/get-started/installation <br/>
     &emsp; - To install PyTorch, run the following command in your terminal: **pip install torch torchvision** <br/>
     &emsp;&emsp;&emsp;For more device-specific instructions on installing torch, refer to this link: https://pytorch.org/get-started/locally/ <br/>
     &emsp; - To install PIL (Pillow), run the following command in your terminal: **pip install Pillow** <br/>
     &emsp;&emsp;&emsp;For more device-specific instructions on installing PIL, refer to this link: https://wp.stolaf.edu/it/installing-pil-pillow-cimage-on-windows-and-mac/ <br/>
     &emsp; - To install the dotenv library, run the following command in your terminal: **pip install python-dotenv** <br/>
     &emsp;&emsp;&emsp;For more information about this library, refer to this link: https://pypi.org/project/python-dotenv/ <br/>

2. Edit the .env file. There are two variables: The model path and the image path. The MODEL_PATH variable should be the absolute path to the model.pth file on your computer. The IMAGE_PATH variable is only used for running the Testing.py file, and should be the absolute path to an image you're interested in finding the classification of.

3. The main part of this project is the Streamlit app. To run the app, open your terminal and navigate to the directory where the App.py file is located. Once in the directory, type the command **streamlit run App.py**. This should open the app in your default web browser.
