# Applied Data Science @ Columbia
## Fall 2021
## Project 1: A "data story" on the history of philosophy


### [Project Description]

Term: Fall 2021
+ Project Title: Clustering/Classification of Philosophical Texts
+ Author: Nikhil Cherukupalli
+ Project summary: This project aims to study clustering/classification problems in a dataset
consisting of philosophy texts to better understand the texts' dependencies
on philosophers and philosophical schools.


### Dependencies
The raw data needs to be dowloaded from: https://www.kaggle.com/kouroshalizadeh/history-of-philosophy
and saved as ``data/raw_data.csv``.

All the library dependencies are listed under 'requirements.txt'. To install all the
dependencies, simply run ```pip install -r requirements.txt``` at the root
of the directory.


### Structure
This directory is organized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```
Please see each subfolder for a README file.

Note that each function in ```lib/``` contains docstrings. To understand 
functionality, simply run ```help(function)``` in any standard Python kernel.
