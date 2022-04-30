import logging

from typing import Optional, Union

from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


class AWSInterface:
    def __init__(self, spark: SparkSession, extraction_bucket: str,
                 sentiment_bucket: str, partitions: int = 32,
                 extraction_date: Optional[Union[str, datetime]] = None,
                 log_level: int = logging.INFO):

        self.logger = logging.getLogger("AWSInterface")
        self.logger.setLevel(log_level)

        self.spark = spark

        self.extraction_bucket = extraction_bucket
        self.sentiment_bucket = sentiment_bucket

        self.partitions = partitions

        if extraction_date is None:
            yesterday = datetime.now() - timedelta(days=1)
            self.extraction_date = yesterday
        else:
            self.extraction_date = extraction_date

    @property
    def extraction_bucket(self) -> str:
        return self.__extraction_bucket

    @extraction_bucket.setter
    def extraction_bucket(self, name: str):
        if type(name) != str:
            raise ValueError("Bucket name is not a string.")

        self.__extraction_bucket = name

    @property
    def sentiment_bucket(self) -> str:
        return self.__sentiment_bucket

    @sentiment_bucket.setter
    def sentiment_bucket(self, name: str):
        if type(name) != str:
            raise ValueError("Bucket name is not a string.")

        self.__sentiment_bucket = name

    @property
    def partitions(self) -> int:
        return self.__partitions

    @partitions.setter
    def partitions(self, n: int):
        if type(n) != int:
            raise ValueError("Partitions is not an integer")
        elif n <= 0:
            raise ValueError("Partitions is not greater than 0.")

        self.__partitions = n

    @property
    def extraction_date(self) -> str:
        return self.__extraction_date

    @extraction_date.setter
    def extraction_date(self, new_date: Union[str, datetime]):
        if type(new_date) == datetime:
            parsed_date = new_date
        elif type(new_date) == str:
            try:
                parsed_date = datetime.fromisoformat(new_date)
            except ValueError as parse_error:
                setter_error = ValueError("Extraction date isn't ISO format.")
                raise setter_error from parse_error
        else:
            raise ValueError("Extraction date is not a string or datetime.")

        if parsed_date > datetime.now():
            raise ValueError("Extraction date is in the future.")

        self.__extraction_date = parsed_date.date().isoformat()

    @property
    def extraction_url(self) -> str:
        return f"s3a://{self.extraction_bucket}/" \
               f"date_crawled={self.extraction_date}/" \
               f"language=en/"

    @property
    def sentiment_url(self) -> str:
        return f"s3a://{self.sentiment_bucket}/"

    def __preprocess_dataframe(self, df: DataFrame) -> DataFrame:
        dates = F.when(df["date_publish"].isNull(), self.extraction_date) \
            .otherwise(df["date_publish"])

        return df.withColumnRenamed("title", "text") \
            .withColumn("date_publish", dates) \
            .withColumn("language", F.lit("en")) \
            .repartition(self.partitions)

    def download(self, limit: Optional[int] = None) -> DataFrame:
        self.logger.info(f"Downloading from '{self.extraction_url}'.")
        df = self.spark.read.parquet(self.extraction_url)

        if limit is not None:
            self.logger.info(f"Reducing dataframe to {limit} rows.")
            df = df.limit(limit)

        self.logger.debug("Setting language to 'en' and null publish"
                          f"dates to '{self.extraction_date}'.")

        return self.__preprocess_dataframe(df)

    def upload(self, df: DataFrame):
        self.logger.info(f"Uploading results to '{self.sentiment_url}'.")
        df.write.mode('append').parquet(self.sentiment_url)

        self.logger.info("Upload successful.")
