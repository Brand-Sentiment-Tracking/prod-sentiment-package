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