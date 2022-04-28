import sparknlp
import logging
import os

from pyspark.sql import SparkSession

from brand_sentiment.awsinterface import AWSInterface
from brand_sentiment.identification import BrandIdentification
from brand_sentiment.sentiment import SentimentIdentification

logging.basicConfig(level=logging.WARN)

extraction_bucket = os.environ.get("EXTRACTION_BUCKET_NAME")
sentiment_bucket = os.environ.get("SENTIMENT_BUCKET_NAME")
extraction_date = os.environ.get("EXTRACTION_DATE")

ner_model_name = "ner_conll_bert_base_cased"
sentiment_model_name = "classifierdl_bertwiki_finance_sentiment_pipeline"

spark = SparkSession.builder \
    .appName("ArticleParquetToDF") \
    .config("spark.sql.broadcastTimeout", "36000") \
    .config("fs.s3.maxConnections", 100) \
    .getOrCreate()

logging.warning(f"Running Apache Spark v{spark.version}")
logging.warning(f"Running Spark NLP v{sparknlp.version()}")

aws_interface = AWSInterface(spark, extraction_bucket, sentiment_bucket,
                             extraction_date)

brand_identifier = BrandIdentification(spark, ner_model_name)
sentimentiser = SentimentIdentification(spark, sentiment_model_name)

articles_df = aws_interface.download()
logging.info(articles_df.shape())
articles_df.show()

brand_spark_df = brand_identifier.predict(articles_df)
logging.info(brand_spark_df.shape())
brand_spark_df.show()

complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
logging.info(complete_spark_df.shape())
complete_spark_df.show()

aws_interface.upload(complete_spark_df)
