# neural-network-challenge-1
Module 18 Challenge

Project Overview

Hey there! This is my Python project for predicting student loan repayment using a neural network. The idea is to use historical student loan data to help predict whether a student is likely to repay their loan. By using machine learning (specifically neural networks), we can make more informed predictions based on student data like credit ranking, income, and other factors.

Since I’m still learning Python (and machine learning in general), this project was a great opportunity to get hands-on experience with building models, training them, and evaluating their performance.

How It Works
In this project, I used TensorFlow to build and train a neural network model. The dataset I used includes information about students' financial backgrounds, and the goal is to predict whether they will repay their loans based on that information.

Here’s a breakdown of the steps I followed:

Data Preparation: I started by loading a CSV file containing data about students. I used Pandas to inspect and prepare the data. The key part was to split the data into features (the things we know about the students) and the target (whether they repaid their loan).

Model Creation: Next, I used TensorFlow's Keras API to create a neural network. I kept it pretty simple—two hidden layers with ReLU activation and one output layer with a sigmoid activation for binary classification (i.e., predicting if a student will or won’t repay their loan).

Training and Evaluation: I trained the model using 50 epochs and validated it using test data. After training, I evaluated the model's performance based on accuracy and loss to see how well it performed on unseen data.

Predictions: Finally, I used the model to predict loan repayment on the test data. The predictions were saved into a DataFrame and compared with the actual outcomes to generate a classification report, showing things like precision, recall, and F1-score.

Files in the Project
student_loans.csv: This is the dataset that contains information about student loans, income, credit scores, etc.

student_loans.keras: This is the trained model saved after fitting the neural network to the training data.

student_loan_prediction.ipynb: This is my Jupyter notebook where all the magic happens! It includes all the code for data preparation, model training, and prediction.

README.md: (This file!) It’s a summary of the project to give you an idea of what I did.


How to Run This Project
If you want to run this project yourself, here’s what you’ll need to do:

1. Install the required libraries:

    1.1 I used the following libraries:
        a. pandas
        b. tensorflow
        c. scikit-learn

You can install them by running:
pip install pandas tensorflow scikit-learn

2. Load the Dataset: Download the dataset or use the one provided in the project (student_loans.csv).

3. Run the Notebook: Open the student_loan_prediction.ipynb notebook in Jupyter or Google Colab. It contains all the steps needed to load the data, preprocess it, build the model, and make predictions.

4. Train and Test the Model: Follow the steps in the notebook to train the model and test it on the dataset. You can also modify things like the number of neurons or epochs if you're feeling adventurous!

5. Make Predictions: Once the model is trained, you can use it to predict whether a new student will repay their loan.


Conclusion
Overall, this project gave me a solid introduction to building and evaluating a neural network in Python. While I’m still learning, I feel proud of what I accomplished, and I’m excited to continue improving my skills. If you're also learning Python, I hope this project inspires you to dive into machine learning—it’s really fun once you get the hang of it!

Thanks for checking out my project!
