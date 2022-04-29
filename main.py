import sparknlp
import logging
import os

from pyspark.sql import SparkSession

from brand_sentiment.awsinterface import AWSInterface
from brand_sentiment.identification import BrandIdentification
from brand_sentiment.sentiment import SentimentIdentification

logging.basicConfig(level=logging.INFO)

extraction_bucket = os.environ.get("EXTRACTION_BUCKET_NAME")
sentiment_bucket = os.environ.get("SENTIMENT_BUCKET_NAME")
extraction_date = os.environ.get("EXTRACTION_DATE")

sentiment_model = os.environ.get("SENTIMENT_MODEL")
ner_model = os.environ.get("NER_MODEL")

dataframe_partitions = int(os.environ.get("DATAFRAME_PARTITIONS"))

spark = SparkSession.builder \
    .appName("ArticleParquetToDF") \
    .config("spark.sql.broadcastTimeout", "36000") \
    .config("fs.s3.maxConnections", 100) \
    .getOrCreate()

logging.warning(f"Running Apache Spark v{spark.version}")
logging.warning(f"Running Spark NLP v{sparknlp.version()}")

aws_interface = AWSInterface(spark, extraction_bucket, sentiment_bucket,
                             extraction_date)

articles_df = aws_interface.download(dataframe_partitions)
shape = (articles_df.count(), len(articles_df.columns))
partitions = articles_df.rdd.getNumPartitions()

logging.warning(f"AWS Download complete with shape {shape} and {partitions} partitions.")
articles_df.show()

brand_identifier = BrandIdentification(spark, ner_model)

brand_df = brand_identifier.predict(articles_df)

logging.info("NER Analysis complete.")
brand_df.show()

sentimentiser = SentimentIdentification(spark, sentiment_model)
brand_sentiment_df = sentimentiser.predict(brand_df)

logging.info(f"Sentiment Analysis complete.")
brand_sentiment_df.show()

aws_interface.upload(brand_sentiment_df)
