# Package imports
import os                 # To handle file paths and directories
import kagglehub          # To download datasets from Kaggle
import pandas as pd       # To read and manipulate the dataset


# <//////////////////////////////////////-- Dataset Initialization --//////////////////////////////////////> 

# Download latest version of dataset
path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")      # Amazon Reviews Dataset

# Get the path to the downloaded kaggle Amazon reviews dataset file 
print("Path to downloaded Amazon CSV dataset files:", path)

# Read the Amazon reviews dataset .CSV file 
amazon_dataset_path = os.path.join(path, "Amazon_Reviews.csv")

# Print the path to the Amazon reviews dataset .CSV file
print("\nPath to Amazon CSV dataset file:", amazon_dataset_path)    

# Transform the Amazon reviews dataset .CSV file into a pandas DataFrame
data = pd.read_csv(
    amazon_dataset_path,
    engine="python",
    nrows=18500            # capped to 18,500 entries
)

# Print some rows of the Amazon reviews dataset
print("\nSome rows of the amazon dataset: \n", data.head())

# Print the info of Amazon reviews dataset 
print("\nPrint dataset info: ")
print(data.info())

# <//////////////////////////////////////-- Dataset Initialization --//////////////////////////////////////> 




# <//////////////////////////////////////-- Dataset Cleaning --//////////////////////////////////////> 

# Removed unnecessary columns from the Amazon reviews dataset
data_dropped_unnecessary_columns = data.drop(["Reviewer Name", "Profile Link", "Country", "Review Count", "Review Date", "Review Title", "Date of Experience"], axis=1)
data = data_dropped_unnecessary_columns

# Print some rows of the Amazon reviews dataset after deleting unnecessary columns
print("\nSome rows after deleting unnecessary columns:", data.head())

# Print the info of Amazon reviews dataset after deleting unnecessary columns
print("\nPrint dataset info after deleting unnecessary columns: ")
print(data.info())

# Left Rating and Review Text columns
# Rating column value is like "Rated 1 out of 5 stars", we will extract the rating value from this string and convert it to an integer
data["Rating"] = data["Rating"].str.extract(r'(\d+)').astype(int)

# Print info and some rows of the Amazon reviews dataset after extracting rating values from the Rating column
print("\nDataset info after extracting rating values:")
print(data.info())
print("\nSome rows after extracting rating values from the Rating column:", data.head())

# Print the average length of every review text in the Review Text column
data["Review Text Length"] = data["Review Text"].apply(len)
print("\nAverage length of review texts:", data["Review Text Length"].mean())

# Print the number of unique values in the Review Text column
print("\nNumber of unique values in the Review Text column:", data["Review Text"].nunique())



# Left Review Text column to clean
# 1. Remove URL 
# 2. Remove puncuation marks
# 3. Remove stop words
# 4. Convert to lowercase (Case folding)
# 5. Lemmatization
# 6. Stemming
# 7. Token Cleaning (includes removing special characters, numbers, and extra whitespace)
# 8. Text abbreviation correction
# 9. Text augmentation
# 10. Tokenization
# 11. Word encoding (vectorization)
# 12. Feature engineering (TDF-IDF vector, and feature extraction like avg word length, avg sentence length, etc.)
# 13. Word Embedding


# <//////////////////////////////////////-- Dataset Cleaning --//////////////////////////////////////> 







# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 
# <//////////////////////////////////////-- Model Building --//////////////////////////////////////> 
