from typing import Dict, Tuple

import pandas as pd
# from pyspark.sql import Column
# from pyspark.sql import DataFrame as SparkDataFrame
# from pyspark.sql.functions import regexp_replace
# from pyspark.sql.types import DoubleType
# from pyspark.sql.functions import udf
# from pyspark.sql.types import StringType

from kedro.io import MemoryDataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # we only need pyplot
sns.set()  # set the default Seaborn style for graphics


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

# Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    return text

# Stemming
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

# preprocess description column on cleaning and stemming
def preprocess_descriptions(train_data: pd.DataFrame):
    # Ensure all values in 'Description' are strings and handle NaN values
    train_data['Description'] = train_data['Description'].astype(str).fillna('')

    # Apply your preprocessing functions
    train_data['Description'] = train_data['Description'].apply(lambda x: remove_stopwords(x))
    train_data['Description'] = train_data['Description'].apply(lambda x: clean_text(x))
    train_data['Description'] = train_data['Description'].apply(stemming)
    
    return train_data

def one_hot_encode_column(df: pd.DataFrame, description_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    unique_values = df[target_col].unique()
    descriptions = df[description_col].unique()
    for description in descriptions:
        row = {'Description': description}
        for unique_value in unique_values:
            row[unique_value] = 1 if unique_value in df[df['Description']
                                                        == description][target_col].values else 0
        rows.append(row)
    new_df = pd.DataFrame(rows, columns=['Description']+list(unique_values))
    return new_df

def create_encoded_df(df: pd.DataFrame, description_col: str, target_col: str):
    unique_departments = df['Department'].unique()
    partitioned_data = {}
    for department in unique_departments:
        df_department = df[df['Department'] == department]
        encoded_df = one_hot_encode_column(df_department, description_col, target_col)
        partitioned_data[department] = encoded_df
    return partitioned_data

def create_department_encoded_df(df: pd.DataFrame) -> dict:
    return one_hot_encode_column(df[['Description', 'Department']], 'Description', 'Department')

def create_techgroup_encoded_df(df: pd.DataFrame) -> dict:
    return create_encoded_df(df, description_col="Description", target_col="Tech Group")

def create_subcategory_encoded_df(df: pd.DataFrame) -> dict:
    return create_encoded_df(df, description_col="Description", target_col="Sub-Category")

def create_category_encoded_df(df: pd.DataFrame) -> dict:
    return create_encoded_df(df, description_col="Description", target_col="Category")


def merge_datasets(department_df: pd.DataFrame, category_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(department_df, category_df,
                         on='Description', how='inner')
    return merged_df

# def _is_true(x: Column) -> Column:
#     return x == "t"


# def _parse_percentage(x: Column) -> Column:
#     x = regexp_replace(x, "%", "")
#     x = x.cast("float") / 100
#     return x


# def _parse_money(x: Column) -> Column:
#     x = regexp_replace(x, "[$£€]", "")
#     x = regexp_replace(x, ",", "")
#     x = x.cast(DoubleType())
#     return x


# def preprocess_companies(companies: SparkDataFrame) -> Tuple[SparkDataFrame, Dict]:
#     """Preprocesses the data for companies.

#     Args:
#         companies: Raw data.
#     Returns:
#         Preprocessed data, with `company_rating` converted to a float and
#         `iata_approved` converted to boolean.
#     """
#     companies = companies.withColumn("iata_approved", _is_true(companies.iata_approved))
#     companies = companies.withColumn("company_rating", _parse_percentage(companies.company_rating))

#     # Drop columns that aren't used for model training
#     companies = companies.drop('company_location', 'total_fleet_count')
#     return companies, {"columns": companies.columns, "data_type": "companies"}


# def load_shuttles_to_csv(shuttles: pd.DataFrame) -> pd.DataFrame:
#     """Load shuttles to csv because it's not possible to load excel directly into spark.
#     """
#     return shuttles


# def preprocess_shuttles(shuttles: SparkDataFrame) -> SparkDataFrame:
#     """Preprocesses the data for shuttles.

#     Args:
#         shuttles: Raw data.
#     Returns:
#         Preprocessed data, with `price` converted to a float and `d_check_complete`,
#         `moon_clearance_complete` converted to boolean.
#     """
#     shuttles = shuttles.withColumn("d_check_complete", _is_true(shuttles.d_check_complete))
#     shuttles = shuttles.withColumn("moon_clearance_complete", _is_true(shuttles.moon_clearance_complete))
#     shuttles = shuttles.withColumn("price", _parse_money(shuttles.price))

#     # Drop columns that aren't used for model training
#     shuttles = shuttles.drop('shuttle_location', 'engine_type', 'engine_vendor', 'cancellation_policy')
#     return shuttles


# def preprocess_reviews(reviews: SparkDataFrame) -> SparkDataFrame:
#     # Drop columns that aren't used for model training
#     reviews = reviews.drop('review_scores_comfort', 'review_scores_amenities', 'review_scores_trip', 'review_scores_crew', 'review_scores_location', 'review_scores_price', 'number_of_reviews', 'reviews_per_month')
#     return reviews


# def create_model_input_table(
#     shuttles: SparkDataFrame, companies: SparkDataFrame, reviews: SparkDataFrame
# ) -> SparkDataFrame:
#     """Combines all data to create a model input table.

#     Args:
#         shuttles: Preprocessed data for shuttles.
#         companies: Preprocessed data for companies.
#         reviews: Raw data for reviews.
#     Returns:
#         Model input table.

#     """
#     # Rename columns to prevent duplicates
#     shuttles = shuttles.withColumnRenamed("id", "shuttle_id")
#     companies = companies.withColumnRenamed("id", "company_id")

#     rated_shuttles = shuttles.join(reviews, "shuttle_id", how="left")
#     model_input_table = rated_shuttles.join(companies, "company_id", how="left")
#     model_input_table = model_input_table.dropna()
#     return model_input_table

