# Disaster_Response_Pipeline_project

## Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Besides the Anaconda distribution of Python, in order to run the machine learning web app, I used the following libraries:
- **plotly**, used for plot on html website, can be installed using `pip install plotly`
- **Flask**, used for manage the back end of the app, can be installed using `pip install Flask`
- The following libraries are needed: sklearn, nltk, sqlalchemy
- **Boostrap** is used for building the web app


## Project Motivation<a name="motivation"></a>

For this project, analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.
[Figure Eight Disaster Dataset](https://www.figure-eight.com/dataset/combined-disaster-response-data/)
- Build a ETL pipeline to clean and prepare the data, and store the cleaned data into a database.
- Builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV and Exports the final model as a pickle file.
- Build a Flask Web App, use the saved machine learning model to classify new texts.

## File Descriptions <a name="files"></a>

There are two notebooks(`ETL Pipeline Preparation.ipynb` and `ML Pipeline Preparation.ipynb` ) available here to showcase work related to the above two pipelines.  This notebooks is exploratory in searching through the data pertaining to the questions showcased by the notebook title.  Markdown cells were used to assist in walking through the thought process for individual steps.  

The `workspace` folder is used for build a local web app. Inside the `workspace` folder, `data` folder contains all datasets and ETL pipeline python file and the database. `models` folder contain ML pipeline python file. `app` folder contains all necessary html and python files to build the web app.

I used the randomforestcalssifier and get a f1 score more than 95.7%. But the pickle file of the model is too big (more than 6G) to upload, anyone use the file should run the `train_classifier.py` to get the pickle file.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@edifierxuhao123/how-to-run-an-airbnb-business-in-sydney-eed9b30d6c40).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to figure eight for the dataset.  
The pipeline and web app files Referenced udacity data-engineer course.

Otherwise, this software is follow a MIT Licenec.

contact email: edifierxuhao123@gmail.com
