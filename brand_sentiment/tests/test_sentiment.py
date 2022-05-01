import unittest

from pyspark.sql import SparkSession

from .. import SentimentIdentification

class TestSentimentIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.spark = SparkSession.builder \
            .appName("TestSentimentIdentification") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.model_name = "classifierdl_bertwiki_finance_sentiment_pipeline"
        self.partitions = 32

        super().__init__(*args, **kwargs)

    def setUp(self):
        self.brand = SentimentIdentification(self.spark, self.model_name,
                                             self.partitions)
        return super().setUp()

    def test_valid_model_name(self):
        pass

    def test_invalid_model_name(self):
        pass

    def test_unknown_model_name(self):
        pass

    def test_valid_partition_size(self):
        pass

    def test_invalid_partition_size(self):
        pass

    def test_predict_sentiment_valid_df(self):
        pass

    def test_predict_sentiment_empty_df(self):
        pass

if __name__ == "__main__":
    unittest.main()