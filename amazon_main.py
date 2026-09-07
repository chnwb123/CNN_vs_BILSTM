# Package imports
import os   
import kagglehub
import pandas as pd

# Download latest version of dataset
path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")

# Get the path to the kaggle downloaded .CSV dataset file 
print("Path to downloaded Amazon CSV dataset files:", path)

# Read the .CSV dataset file into a pandas DataFrame
amazon_dataset_path = os.path.join(path, "Amazon_Reviews.csv")
print("Path to Amazon CSV dataset file:", amazon_dataset_path)
data = pd.read_csv(
    amazon_dataset_path,
    engine="python",
    nrows=15000
)
print("Print some rows of the amazon dataset: \n", data.head())
print("Print dataset info: \n")
print(data.info())