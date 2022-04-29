from datetime import date, timedelta

from pyspark.sql import SparkSession, DataFrame, functions as F

class AWSInterface:
    def __init__(self, spark: SparkSession, extraction_bucket: str,
                 sentiment_bucket: str, extraction_date: str):

        self.spark = spark

        self.extraction_bucket = extraction_bucket
        self.sentiment_bucket = sentiment_bucket

        if extraction_date is None:
            yesterday = date.today() - timedelta(days=1)
            self.extraction_date = yesterday.isoformat()
        else:
            self.extraction_date = extraction_date

    def __preprocess_dataframe(self, df: DataFrame,
                               partitions: int) -> DataFrame:

        dates = F.when(df["date_publish"].isNull(), self.extraction_date) \
            .otherwise(df["date_publish"])

        return df.withColumnRenamed("title", "text") \
            .withColumn("date_publish", dates) \
            .withColumn("language", F.lit("en")) \
            .repartition(partitions)

    def download(self, partitions: int = 32) -> DataFrame:
        df = self.spark.read \
            .parquet(f"s3a://{self.extraction_bucket}/"
                     f"date_crawled={self.extraction_date}/"
                     f"language=en/")

        return self.__preprocess_dataframe(df, partitions)

    def upload(self, df: DataFrame):
        df.write.mode('append').parquet(f"s3a://{self.sentiment_bucket}/")

    def save_locally(self, df: DataFrame):
        df.write.csv('/tmp/output')
