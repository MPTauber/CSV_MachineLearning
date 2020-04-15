## py -3 -m venv nlp_venv
## nlp_venv/scripts/activate
## pip install pandas

import pandas as pd 

# Make formula that automatically reads files and converts into dataframes
def csv_to_df(file):
    df = pd.read_csv(file)
    return pd.DataFrame(df)

animal_classes = csv_to_df("animal_classes.csv")
animals_test = csv_to_df("animals_test.csv")
animals_train = csv_to_df("animals_train.csv")
predictions = csv_to_df("predictions.csv")

