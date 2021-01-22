## Activity Recognition from a Single Chest-Mounted Accelerometer

## Objective
1. Create a Supervised Learning ML model that can accept raw values from a 3 DoF sensor 
2. Classify up to 5 different(classes) activities based on the raw data. 
3. Make sure to include a readme file justifying the various steps being performed as well as why the ML model you’re using was selected. (The readme file is mandatory for submissions to be considered.) 
4. Submissions are to be made in the form of a GitHub repo with a requirements.txt file included. 

## Abstract and Relevant Information (Given) 
The dataset collects data from a wearable accelerometer mounted on the chest. 
Uncalibrated Accelerometer Data are collected from 15 participants performing 7 activities. 
The dataset is intended for Activity Recognition research purposes. 
It provides challenges for identification and authentication of people using motion patterns.
The dataset collects data from a wearable accelerometer mounted on the chest.
Sampling frequency of the accelerometer: 52 Hz
Accelerometer Data are Uncalibrated
Number of Participants: 15
Number of Activities: 7
Data Format: CSV

## Pre-Requirements
### Essential Libraries and Tools
<b>Python</b> programming language is used to complete this project. <b>version used: 3.6.5</b>.
The Essential Libraries used in this project are <b>os, pickle</b> for file handling, <b>numpy and pandas</b> for data analysis, <b>matplotlib and seaborn</b> for data insight and visualization, <b>scikit-learn</b> for using machine learning models and <b>jupyter notebook</b> for documentation. Please refer to the requirement.txt file for all the version details of the libraries.
 
## Dataset
### Data Preparation
The collected data of 15 participants were given in fifteen .csv(comma separated value) format files for the project.
Initially, all the .csv files was kept in a folder named 'data'.
From this 'data' folder, all the files were merged in a single pandas dataframe. The first column i.e. "sequential number" was replaced by 'filename' while generating the dataframe which was containing the name of the respective files (eg. 1.csv) of the data. This master file is also saved in a foleder named "model" as "dataset.csv".
The merged dataset contains 1926896 rows and 5 columns.
Columns Description : filename, x acceleration, y acceleration, z acceleration, label.
Here, the x acceleration, y acceleration,  z acceleration, represents pitch, roll and yaw respectively. 
Pitch represents the up and down movement, Roll represents left and right movement and Yaw represents the clockwise and anticlockwise movement.

### Data Cleaning
According to the given file description the Labels were codified by numbers from 1 to 7. Each number representing an activity like Working at Computer,Standing Up, Walking and Going up\down stairs, Standing, Walking, Going Up\Down Stairs, Walking and Talking with Someone, Talking while Standing.
In the label column, another class apart from 1 to 7 named '0' was observed.
The value count of '0' was 3719 which was removed from the dataset.
It is also observed that there was no "Null" value in the entire dataset.
### Data Visualization
The data of the columns x acceleration, y acceleration, z acceleration was plotted individually as distribution plot. A bell curve is observed in each case which signifies the data follows normal distribution and is ready to be used for machine learning model training purpose.
### Feature Engineering
The sampling frequency of the accelerometer is 52 Hz. It means that 52 records was fetched in one second. Additionally it is mentioned that the data is uncalibrated i.e. not corrected or adjusted so there are possibilities of contating outliers in the data. Initially, the data x, y and z acceleration was plotted as boxplot and it was observed that the dataset was containing outliers. Hence we have used Interquartile range and removed the unwanted data.

## Model Development
The prepared experimental dataset is split into two parts as training(80%) and testing(20%) datasets.
Since the model is based on classification problem and predicts a discrete output of Activity Recognition(Working at Computer,Standing Up, Walking and Going up\down stairs, Standing, Walking, Going Up\Down Stairs, Walking and Talking with Someone, Talking while Standing), we have used classification algorithms namely Decision Tree, Gaussian Naïve Bayes and K-Nearest Neighbours (KNN) algorithm to build our initial model. 

### Initial Model Accuracy
<b>MODELNAME	          ACCURACY</b>
<li>DecisionTree-train:	0.956270</li>
<li>DecisionTree-test:	  0.658319</li>
<li>KNN-train(5):        0.790427</li>
<li>KNN-test(5):         0.724527<l/i>
<li>NB-train:	          0.433865</li>
<li>NB-test:	            0.433233</li>
It is clearly observed in case of decision tree the model works good in the training data but gives inadequate output in the testing set. The Naive Bayes model fails to produce proper output.  
K-Nearest Neighbour algorithm with k value as 5 gives the best output among the used models.

### Selecting the model and Increasing the model accuracy
In the dataset it is observed that in the 'label' column there are repetative classes like,
2: "Standing Up, Walking and Going up\down stairs" is a merged class of 3,4 and 5. 
We have checked the value counts for each classes and decided to delete the lower count classes i.e., 5,2 and 6.
<li>1:    608667</li>
<li>7:    593563</li>
<li>4:    357064</li>
<li>3:    216737</li>
<li>5:     51498</li>
<li>2:     47878</li>
<li>6:     47770</li>
Name: 4, dtype: int64
Now, The selected classifier, the KNN model is used and the range for the value of K is taken from 15 to 50 to select the best value of K.
Although the training and testing accuracy was promising (in most cases, training : 82%, testing : 80%) but the Error Rate was fluctuating from K value 29 to 50.
So, we decided to add another feature column containing the average of x acceleration, y acceleration and z acceleration for each rows.
Finally, another column was added which contains the magnitude of acceleration: Am = root(x^2+y^2+z^2) 

After this, we fit our data to the selected KNN model ranging the value of K from 5 to 50, with intervals of 2 which resulted a smooth Error Graph and a promising training and testing accuracy. 
We finally decided to select the value of K as 29 for our final model, since the values after that subtly starts to fluctuate. The final model gives accuracy as training data: 0.817491, testing:	0.809497 which increased by 5% from the last model(where all the label classes were included) and an error rate of 0.19053. Our final model can predict 4 acitivities (Working at Computer, Standing, Walking and Talking while Standing). This model is saved as a pickle file in the "models" folder as <b>"final_model.sav"</b> is the required machine learning model.

## Conclusion
This project was aimed at designing a model which would predict the 5 different activities based on the raw data. For developing this model, we applied data cleaning and feature engineering on the given dataset in order to simplify the data. After that, the unnecessary classes from the label field were removed, values on the basis of how they were affecting our output.
After that, the datasets were subdivided into training and test data and different algorithms like Naive Bayes, Decision Tree, K-Nearest Neighbours were used to determine which
algorithm gave way to the most precise model. After applying a few algorithms on the dataset,it was observed that <b>KNN algorithm with K value 29 gave the best results as follows:
  <li>Training Accuracy: 0.817491</li>
  <li>Testing Accuracy: 0.809497</li> 
</b>
Please refer to the two jupyter notebooks named as "DataAnalysis&Viz" and "ModelSelection" in the "notebooks" folder for all the data analysis, data visualization graphs and model informations.

## Instructions
<li>Go to the root folder and open a command terminal</li>
<li>type the command pip install -r requirement.txt(windows users only)</li>
<li>type the command python main.py </li>
<li>Please refer to the notebook folder "Updates" and "FeatureEngineering" files for the updated assignment solutions</li>
