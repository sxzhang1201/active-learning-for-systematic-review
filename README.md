
# Active learning for Workload Reduction of Automated Screening in Systematic Reviews

This repository contains the code that implements the research described in Shuxin Zhang's master thesis. He carried out this work for his Master Medical Informatics at the University of Amsterdam. The project was supervised by Sílvia Delgado Olabarriaga (tutor) and Allard van Altena (mentor) from the Department of Clinical Epidemiology, Biostatistics, and Bioinformatics, Amsterdam University Medical Center. With special thanks to the CLEF eHealth Lab for providing the necessary data.

There are four major parts (four folders) of Python code which were implemented for this project. These correspond to the four chapters in the thesis:

Chapter 3 Data Preparation

Chapter 4 Baseline Model Design

Chapter 5 Active Learning Model Design

Chapter 6 Evaluation of Active Learning Moddel

### Note (January 2019)
run_baseline.py, run_similarity.py, and result_analysis.py are updated codes for further investigation after thesis project. 


## Getting Started 

### Prerequisites

The whole project is implemented in Python. The required packages are listed in requirement.txt. 

### Installing

With Python and pip installed you can run the following command to install the dependencies:

$pip install -r requirements.txt

## Running

To get the results of all experiments, you should execute the following Python Scripts step-by-step:

#### 1. parse_example.py and run_data_preparation.py in 'data_preparation' folder

This creates: a SQL database in the folder 'Database'; fifty pickle files (forty full datasets and ten partial datasets) in the folder 'Datasets'; and one 'word frequency' graph in the folder 'Figure'.
##### fetch_raw.py
Loop the qrel files (train and test) and fetch the pubmed articles, stick them in a pickle file.

##### insert_docs.py
Read the pickle files created by fetch_raw.py and move them to the database.

##### fetch_validity
Checks database against the qrel files and checks whether everything was successfully fetched and stored.

##### clean docs
Run preprocessing on the documents and store as a feature matrix.

#### 2. run_baseline_design.py in 'baseline_model' folder

This part corresponds to all experiments described in Chapter 4 Baseline Model Design of the thesis.

All results are stored as pickle files in the folder "Consolidation", whereas graphs are stored in the 'Figure' folder.

#### 3. run_active_learning.py in 'active_learning_model' folder

This part corresponds to all experiments described in Chapter 5 Active Learning Model Design of the thesis.

All results are stored as pickle files in the folder "Consolidation", whereas graphs are stored in the 'Figure' folder.

#### 4. run_added_value.py in 'added_value' folder

This part corresponds to all experiments described in Chapter 6 Evaluation of Active Learning Moddel of the thesis.

All results are stored as pickle files in the folder "Consolidation", whereas graphs are stored in the 'Figure' folder.

#### Note

config.py describes the storage 'path' of results and graphs for most experiments;

recipe.py describes which value is selected for each parameter in the classifier. This file should be edited according to the actual experiment results.



## Authors: 
Shuxin Zhang, Medical Informatics Master student,  University of Amsterdam.

Allard van Altena, PhD candidate, University of Amsterdam.


## Acknowledgement
Sílvia Delgado Olabarriaga (tutor), assistant professor, University of Amsterdam.

[CLEF eHealth Lab](http://clef2017.clef-initiative.eu/)

