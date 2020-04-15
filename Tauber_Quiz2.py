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
print(merged_train)