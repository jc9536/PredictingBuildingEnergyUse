# Predicting Building Energy Use
---
Author: Jaimie Chin
Date: February 9th, 2023
Assignment: DS.301 Assignment 2
---
[Kaggle](www.kaggle.com) is a hub for data science. You can find data sets, code examples, and competitions there. If you don't have one already, make a kaggle account for yourself. 

Read the Overview and Data pages for the WiDS 2022 competition on estimating building energy use: https://www.kaggle.com/competitions/widsdatathon2022/data 

Then download the training data (train.csv)

From that file, plot histograms of values for the following columns:

* floor area

* elevation

* cooling degree days

* heating degree days

* site_eui

```{python}
#Import packages & libraries 
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
```
