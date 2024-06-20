from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

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


def _is_true(x: Column) -> Column:
    return x == "t"


def _parse_percentage(x: Column) -> Column:
    x = regexp_replace(x, "%", "")
    x = x.cast("float") / 100
    return x


def _parse_money(x: Column) -> Column:
    x = regexp_replace(x, "[$£€]", "")
    x = regexp_replace(x, ",", "")
    x = x.cast(DoubleType())
    return x


def preprocess_companies(companies: SparkDataFrame) -> Tuple[SparkDataFrame, Dict]:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies = companies.withColumn("iata_approved", _is_true(companies.iata_approved))
    companies = companies.withColumn("company_rating", _parse_percentage(companies.company_rating))

    # Drop columns that aren't used for model training
    companies = companies.drop('company_location', 'total_fleet_count')
    return companies, {"columns": companies.columns, "data_type": "companies"}


def load_shuttles_to_csv(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Load shuttles to csv because it's not possible to load excel directly into spark.
    """
    return shuttles


def preprocess_shuttles(shuttles: SparkDataFrame) -> SparkDataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles = shuttles.withColumn("d_check_complete", _is_true(shuttles.d_check_complete))
    shuttles = shuttles.withColumn("moon_clearance_complete", _is_true(shuttles.moon_clearance_complete))
    shuttles = shuttles.withColumn("price", _parse_money(shuttles.price))

    # Drop columns that aren't used for model training
    shuttles = shuttles.drop('shuttle_location', 'engine_type', 'engine_vendor', 'cancellation_policy')
    return shuttles


def preprocess_reviews(reviews: SparkDataFrame) -> SparkDataFrame:
    # Drop columns that aren't used for model training
    reviews = reviews.drop('review_scores_comfort', 'review_scores_amenities', 'review_scores_trip', 'review_scores_crew', 'review_scores_location', 'review_scores_price', 'number_of_reviews', 'reviews_per_month')
    return reviews


def create_model_input_table(
    shuttles: SparkDataFrame, companies: SparkDataFrame, reviews: SparkDataFrame
) -> SparkDataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    # Rename columns to prevent duplicates
    shuttles = shuttles.withColumnRenamed("id", "shuttle_id")
    companies = companies.withColumnRenamed("id", "company_id")

    rated_shuttles = shuttles.join(reviews, "shuttle_id", how="left")
    model_input_table = rated_shuttles.join(companies, "company_id", how="left")
    model_input_table = model_input_table.dropna()
    return model_input_table

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
    train_data['Description'] = train_data['Description'].apply(
        lambda x: remove_stopwords(x))
    train_data['Description'] = train_data['Description'].apply(
        lambda x: clean_text(x))
    train_data['Description'] = train_data['Description'].apply(stemming)
    return train_data

def one_hot_encode_column(df, description_col, target_col):
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

def create_department_df(df):
    return one_hot_encode_column(df[['Description', 'Department']], 'Description', 'Department')

import os 

def create_techgroup_df(df):
    unique_departments = df['Department'].unique()

    # Create output directory if it does not exist
    output_dir = "data/02_intermediate/techgroup_encoded_dir"
    os.makedirs(output_dir, exist_ok=True)

    for department in unique_departments:
        df_department = df[df['Department'] == department]        
        # Encode tech groups for the current department
        encoded_df = one_hot_encode_column(df_department[['Description', 'Tech Group']], 'Description', 'Tech Group')
    
        # Save encoded DataFrame to CSV
        output_filepath = os.path.join(output_dir, f"{department}_encoded.csv")
        encoded_df.to_csv(output_filepath, index=False)

    return output_dir

def create_category_df(df):
    unique_departments = df['Department'].unique()
    output_dir = "data/02_intermediate/category_encoded_dir"
    os.makedirs(output_dir, exist_ok=True)

    for department in unique_departments:
        df_department = df[df['Department'] == department]
        encoded_df = one_hot_encode_column(df_department[['Description', 'Category']], 'Description', 'Category')
        # Save encoded DataFrame to CSV
        output_filepath = os.path.join(output_dir, f"{department}_encoded.csv")
        encoded_df.to_csv(output_filepath, index=False)
    return output_dir

def create_subcategory_df(df):
    unique_departments = df['Department'].unique()
    output_dir = "data/02_intermediate/subcategory_encoded_dir"
    os.makedirs(output_dir, exist_ok=True)

    for department in unique_departments:
        df_department = df[df['Department'] == department]
        encoded_df = one_hot_encode_column(df_department[['Description', 'Sub Category']], 'Description', 'Sub Category')
        # Save encoded DataFrame to CSV
        output_filepath = os.path.join(output_dir, f"{department}_encoded.csv")
        encoded_df.to_csv(output_filepath, index=False)
    return output_dir

