import unittest

from pyspark.sql import SparkSession
from datetime import datetime

from .. import AWSInterface

class TestAWSInterface(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.spark = SparkSession.builder \
            .appName("TestAWSInterface") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.unit_test_bucket = "brand-sentiment-unit-testing"

        self.extraction_bucket = f"{self.unit_test_bucket}/downloading"
        self.sentiment_bucket = f"{self.unit_test_bucket}/uploading"

        self.partitions = 32
        self.extraction_date = datetime(2022, 4, 26)

        super().__init__(*args, **kwargs)

    def setUp(self):
        self.interface = AWSInterface(self.spark, self.extraction_bucket,
                                      self.sentiment_bucket, self.partitions,
                                      self.extraction_date)

        return super().setUp()
    
    def test_valid_extraction_bucket(self):
        pass

    def test_invalid_extraction_bucket(self):
        pass

    def test_valid_sentiment_bucket(self):
        pass

    def test_invalid_sentiment_bucket(self):
        pass

    def test_valid_partition_size(self):
        pass

    def test_invalid_partition_size(self):
        pass

    def test_valid_extraction_date_string(self):
        pass

    def test_valid_extraction_date_datetime(self):
        pass

    def test_invalid_extraction_date_bad_type(self):
        pass

    def test_invalid_extraction_date_malformed_string(self):
        pass

    def test_invalid_extraction_date_bad_day(self):
        pass

    def test_extraction_bucket_partition_url(self):
        pass

    def test_sentiment_bucket_url(self):
        pass

    def test_download_parquet_partition_to_spark(self):
        pass

    def test_upload_spark_dataframe_to_parquet(self):
        pass


if __name__ == "__main__":
    unittest.main()