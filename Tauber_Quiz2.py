## py -3 -m venv nlp_venv
## nlp_venv/scripts/activate
## pip install pandas

#### INSTRUCTIONS:
#MACHINE LEARNING SUPERVISED CLASSIFICATION MODEL
# You are to apply skills you have acquired in Machine Learning to correctly predict the classification of a group of animals. 
# The data has been divided into 3 files.

# Classes.csv 
# Is a file describing the class an animal belongs to as well as the name of the class. 
# The class number and class type are the two values that are of most importance to you.

# animals_train.csv 
# Is the file you will use to train your model. There are 101 samples with 17 features. 
# The last feature is the class type (which is actually the class number 1,2,3, etc.). This should be used as your target attribute. 
# However, we want the target attribute to be the class type (Mammal, Bird, Reptile, etc.) instead of the class number (1,2,3,etc.).

# animals_test.csv
# Is the file you will use to test your model to see if it can correctly predict the class that each sample belongs to. 
# The first column in this file has the name of the animal (which is not in the training file).  
# Also, this file does not have a target attribute since the model should predict the target class.

# Your program should produce a csv file that shows the name of the animal and their corresponding class as shown in this file 
# predictions.csv

import pandas as pd 

# Make formula that automatically reads files and converts into dataframes
def csv_to_df(file):
    df = pd.read_csv(file)
    return pd.DataFrame(df)

animal_classes = csv_to_df("animal_classes.csv")
animals_test = csv_to_df("animals_test.csv")
animals_train = csv_to_df("animals_train.csv")
predictions = csv_to_df("predictions.csv")


### Change target attribute class_type in animals_train from Number to Name
merged_train = animals_train.join(animal_classes.set_index("Class_Number"),on='class_type')

### Clean new dataframe
merged_train = merged_train.drop(['Number_Of_Animal_Species_In_Class', 'Animal_Names'], axis=1)
merged_train = merged_train.rename(columns = {"class_type":"class_number", "Class_Type":"class_name"})

### Splitting data for training and testing
from sklearn.model_selection import train_test_split

x = merged_train.drop(['class_number', 'class_name'], axis=1)
y = merged_train["class_name"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 777)


### Training Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=x_train, y=y_train)

predicted = knn.predict(X=x_test)
expected = y_test

#print(predicted[:20])
#print(expected[:20])

# Checking how many wrong:
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p !=e]


## Using model on animal_test

animals_test2 = animals_test.drop(["animal_name"], axis=1)
test = knn.predict(X= animals_test2)


# Making two series out of the test names and the predictions
prediction = pd.Series(test)
animal_name = pd.Series(animals_test["animal_name"])

## Saving the two in the outcome dataframe

Result_animals_prediction = pd.concat([animal_name, prediction], axis=1)

# Save as csv

Result_animals_prediction.to_csv("Tauber_Results_Animal_Predictions.csv")