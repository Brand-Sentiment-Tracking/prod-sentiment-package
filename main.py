import sparknlp
import logging

from os import environ

from brand_sentiment.awsinterface import AWSInterface
from brand_sentiment.identification import BrandIdentification
from brand_sentiment.sentiment import SentimentIdentification

logging.basicConfig(level=logging.WARN)

extraction_bucket_name = environ.get("EXTRACTION_BUCKET_NAME")
sentiment_bucket_name = environ.get("SENTIMENT_BUCKET_NAME")
parquet_filepath = environ.get("PARQUET_FILEPATH")
date = environ.get("EXTRACTION_DATE")

spark = sparknlp.start()

logging.warning(f"Running Apache Spark version {spark.version}")
logging.warning(f"Running Spark NLP version {sparknlp.version()}")

aws_interface = AWSInterface(extraction_bucket_name, sentiment_bucket_name, parquet_filepath, date)
brand_identifier = BrandIdentification("ner_dl_bert")
sentimentiser = SentimentIdentification("custom_pipeline")

logging.warning(spark.conf.get("spark.executor.memory"))

articles_df = aws_interface.s3_parquet()
print(articles_df.head())
brand_spark_df = brand_identifier.predict_brand(articles_df)
print(brand_spark_df.head())
complete_spark_df = sentimentiser.predict_dataframe(brand_spark_df)
print(complete_spark_df.head())
aws_interface.upload(complete_spark_df)
