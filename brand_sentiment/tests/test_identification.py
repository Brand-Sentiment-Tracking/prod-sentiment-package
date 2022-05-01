import unittest

from pyspark.sql import SparkSession

from .. import BrandIdentification

class TestBrandIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.spark = SparkSession.builder \
            .appName("TestBrandIdentification") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.model_name = "xlnet_base"
        self.partitions = 32

        super().__init__(*args, **kwargs)

    def setUp(self):
        self.brand = BrandIdentification(self.spark, self.model_name,
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

    def test_build_xlnet_pipeline(self):
        pass

    def test_build_conll_pipeline(self):
        pass

    def test_predict_brand_xlnet_base(self):
        pass

    def test_predict_brand_conll_bert(self):
        pass


if __name__ == "__main__":
    unittest.main()