# This file was originally taken from TF official repo: https://github.com/tensorflow/models/blob/master/official/datasets/movielens.py
# It has been slightly modified for our usecase.
# ==============================================================================
"""Download and extract the MovieLens dataset from GroupLens website.
Download the dataset, and perform basic preprocessing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile
import zipfile

# pylint: disable=g-bad-import-order
import numpy as np
import pandas as pd
import six
from six.moves import urllib  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + os.sep + os.pardir)
upper_project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + os.sep + os.pardir + os.sep + os.pardir)
sys.path.append(upper_project_dir)

from lucrl.utils.coordinator import Coordinator
crd = Coordinator(project_dir)

from lucrl.utils.logger import Logger
logger = Logger(project_dir, 'download_movielens', 'download_movielens.txt')

ML_1M = "ml-1m"
ML_20M = "ml-20m"
DATASETS = [ML_1M]

RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"
USERS_FILE = "users.csv"

# URL to download dataset
_DATA_URL = "http://files.grouplens.org/datasets/movielens/"

GENRE_COLUMN = "genres"
ITEM_COLUMN = "item_id"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "user_id"

GENDER_COLUMN = 'gender'
AGE_COLUMN = 'age'
OCCUPATION_COLUMN = 'occupation'
ZIP_CODE_COLUMN = 'zipcode'

GENRES = [
    'Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX", 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
N_GENRE = len(GENRES)

OCCUPATION_DICT = {0: "other",
                   1: "academic/educator",
                   2: "artist",
                   3: "clerical/admin",
                   4: "college/grad student",
                   5: "customer service",
                   6: "doctor/health care",
                   7: "executive/managerial",
                   8: "farmer",
                   9: "homemaker",
                   10: "K-12 student",
                   11: "lawyer",
                   12: "programmer",
                   13: "retired",
                   14: "sales/marketing",
                   15: "scientist",
                   16: "self-employed",
                   17: "technician/engineer",
                   18: "tradesman/craftsman",
                   19: "unemployed",
                   20: "writer"}

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]

# UserID::Gender::Age::Occupation::Zip-code
USER_COLUMNS = [USER_COLUMN, GENDER_COLUMN, AGE_COLUMN, OCCUPATION_COLUMN, ZIP_CODE_COLUMN]

# Note: Users are indexed [1, k], not [0, k-1]
NUM_USER_IDS = {
    ML_1M: 6040,
    ML_20M: 138493,
}

# Note: Movies are indexed [1, k], not [0, k-1]
# Both the 1m and 20m datasets use the same movie set.
NUM_ITEM_IDS = 3952

MAX_RATING = 5

NUM_RATINGS = {
    ML_1M: 1000209,
    ML_20M: 20000263
}

def _download_and_clean(dataset, data_dir):
    """Download MovieLens dataset in a standard format.
    This function downloads the specified MovieLens format and coerces it into a
    standard format. The only difference between the ml-1m and ml-20m datasets
    after this point (other than size, of course) is that the 1m dataset uses
    whole number ratings while the 20m dataset allows half integer ratings.
    """
    if dataset not in DATASETS:
        raise ValueError("dataset {} is not in {{{}}}".format(
            dataset, ",".join(DATASETS)))

    data_subdir = os.path.join(data_dir, dataset)

    expected_files = ["{}.zip".format(dataset), RATINGS_FILE, MOVIES_FILE, USERS_FILE]

    tf.gfile.MakeDirs(data_subdir)
    if set(expected_files).intersection(
            tf.gfile.ListDirectory(data_subdir)) == set(expected_files):
        logger.info("Dataset {} has already been downloaded".format(dataset))
        return

    url = "{}{}.zip".format(_DATA_URL, dataset)

    temp_dir = tempfile.mkdtemp()
    try:
        zip_path = os.path.join(temp_dir, "{}.zip".format(dataset))
        zip_path, _ = urllib.request.urlretrieve(url, zip_path)
        statinfo = os.stat(zip_path)
        # A new line to clear the carriage return from download progress
        logger.info(
            "Successfully downloaded {} {} bytes".format(
                zip_path, statinfo.st_size))

        zipfile.ZipFile(zip_path, "r").extractall(temp_dir)

        if dataset == ML_1M:
            _regularize_1m_dataset(temp_dir)
        else:
            _regularize_20m_dataset(temp_dir)

        for fname in tf.gfile.ListDirectory(temp_dir):
            if not tf.gfile.Exists(os.path.join(data_subdir, fname)):
                tf.gfile.Copy(os.path.join(temp_dir, fname),
                              os.path.join(data_subdir, fname))
            else:
                logger.info("Skipping copy of {}, as it already exists in the "
                            "destination folder.".format(fname))

    finally:
        tf.gfile.DeleteRecursively(temp_dir)


def _transform_csv(input_path, output_path, names, skip_first, separator=","):
    """Transform csv to a regularized format.
    Args:
      input_path: The path of the raw csv.
      output_path: The path of the cleaned csv.
      names: The csv column names.
      skip_first: Boolean of whether to skip the first line of the raw csv.
      separator: Character used to separate fields in the raw csv.
    """
    if six.PY2:
        names = [n.decode("utf-8") for n in names]

    with tf.gfile.Open(output_path, "wb") as f_out, \
            tf.gfile.Open(input_path, "rb") as f_in:

        # Write column names to the csv.
        f_out.write(",".join(names).encode("utf-8"))
        f_out.write(b"\n")
        for i, line in enumerate(f_in):
            if i == 0 and skip_first:
                continue  # ignore existing labels in the csv

            line = line.decode("utf-8", errors="ignore")
            fields = line.split(separator)
            if separator != ",":
                fields = ['"{}"'.format(field) if "," in field else field
                          for field in fields]
            f_out.write(",".join(fields).encode("utf-8"))


def _regularize_1m_dataset(temp_dir):
    """
    ratings.dat
      The file has no header row, and each line is in the following format:
      UserID::MovieID::Rating::Timestamp
        - UserIDs range from 1 and 6040
        - MovieIDs range from 1 and 3952
        - Ratings are made on a 5-star scale (whole-star ratings only)
        - Timestamp is represented in seconds since midnight Coordinated Universal
          Time (UTC) of January 1, 1970.
        - Each user has at least 20 ratings
    movies.dat
      Each line has the following format:
      MovieID::Title::Genres
        - MovieIDs range from 1 and 3952
    """
    working_dir = os.path.join(temp_dir, ML_1M)

    _transform_csv(
        input_path=os.path.join(working_dir, "ratings.dat"),
        output_path=os.path.join(temp_dir, RATINGS_FILE),
        names=RATING_COLUMNS, skip_first=False, separator="::")

    _transform_csv(
        input_path=os.path.join(working_dir, "movies.dat"),
        output_path=os.path.join(temp_dir, MOVIES_FILE),
        names=MOVIE_COLUMNS, skip_first=False, separator="::")

    _transform_csv(
        input_path=os.path.join(working_dir, "users.dat"),
        output_path=os.path.join(temp_dir, USERS_FILE),
        names=USER_COLUMNS, skip_first=True, separator="::")

    tf.gfile.DeleteRecursively(working_dir)


def _regularize_20m_dataset(temp_dir):
    """
    ratings.csv
      Each line of this file after the header row represents one rating of one
      movie by one user, and has the following format:
      userId,movieId,rating,timestamp
      - The lines within this file are ordered first by userId, then, within user,
        by movieId.
      - Ratings are made on a 5-star scale, with half-star increments
        (0.5 stars - 5.0 stars).
      - Timestamps represent seconds since midnight Coordinated Universal Time
        (UTC) of January 1, 1970.
      - All the users had rated at least 20 movies.
    movies.csv
      Each line has the following format:
      MovieID,Title,Genres
        - MovieIDs range from 1 and 3952
    """
    working_dir = os.path.join(temp_dir, ML_20M)

    _transform_csv(
        input_path=os.path.join(working_dir, "ratings.csv"),
        output_path=os.path.join(temp_dir, RATINGS_FILE),
        names=RATING_COLUMNS, skip_first=True, separator=",")

    _transform_csv(
        input_path=os.path.join(working_dir, "movies.csv"),
        output_path=os.path.join(temp_dir, MOVIES_FILE),
        names=MOVIE_COLUMNS, skip_first=True, separator=",")
    """
    _transform_csv(
        input_path=os.path.join(working_dir, "users.csv"),
        output_path=os.path.join(temp_dir, USERS_FILE),
        names=USER_COLUMNS, skip_first=True, separator=",")
    """
    tf.gfile.DeleteRecursively(working_dir)


def download(dataset, data_dir):
    if dataset:
        _download_and_clean(dataset, data_dir)
    else:
        _ = [_download_and_clean(d, data_dir) for d in DATASETS]


def ratings_csv_to_dataframe(data_dir, dataset):
    with tf.gfile.Open(os.path.join(data_dir, dataset, RATINGS_FILE)) as f:
        return pd.read_csv(f, encoding="utf-8")


def csv_to_joint_dataframe(data_dir, dataset):
    ratings = ratings_csv_to_dataframe(data_dir, dataset)

    with tf.gfile.Open(os.path.join(data_dir, dataset, MOVIES_FILE)) as f:
        movies = pd.read_csv(f, encoding="utf-8")

    df = ratings.merge(movies, on=ITEM_COLUMN)
    df[RATING_COLUMN] = df[RATING_COLUMN].astype(np.float32)

    return df


def integerize_genres(dataframe):
    """Replace genre string with a binary vector.
    Args:
      dataframe: a pandas dataframe of movie data.
    Returns:
      The transformed dataframe.
    """
    def _map_fn(entry):
        entry.replace("Children's", "Children")  # naming difference.
        movie_genres = entry.split("|")
        output = np.zeros((len(GENRES),), dtype=np.int64)
        for i, genre in enumerate(GENRES):
            if genre in movie_genres:
                output[i] = 1
        return output

    dataframe[GENRE_COLUMN] = dataframe[GENRE_COLUMN].apply(_map_fn)

    return dataframe

def occupation_ids_to_name(dataframe):
    """Replace occupation ids with a string name
    Args:
      dataframe: a pandas dataframe of users data.
    Returns:
      The transformed dataframe.
    """

    dataframe[OCCUPATION_COLUMN] = dataframe[OCCUPATION_COLUMN].replace(OCCUPATION_DICT)

    return dataframe

def define_data_download_flags():
    """Add flags specifying data download arguments."""
    flags.DEFINE_string(
        name="data_dir", default=crd.data_raw.path,
        help="Directory to download and extract data.")

    flags.DEFINE_enum(
        name="dataset", default=None,
        enum_values=DATASETS, case_sensitive=False,
        help="Dataset to be trained and evaluated.")


def main(_):
    """Download and extract the data from GroupLens website."""
    download(flags.FLAGS.dataset, flags.FLAGS.data_dir)

if __name__ == "__main__":
    define_data_download_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)