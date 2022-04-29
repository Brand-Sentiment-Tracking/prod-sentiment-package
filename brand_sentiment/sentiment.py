import logging

from typing import List

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, ArrayType
from sparknlp.pretrained import PretrainedPipeline


class SentimentIdentification:

    SENTIMENT_FIELDS = ("text", "source_domain", "date_publish", "language",
                        "Predicted_Entity", "class.result")

    def __init__(self, spark: SparkSession, model_name: str):
        """Creates a class for sentiment identication using specified model.

        Args:
          model_name: Name of the Spark NLP pretrained pipeline.
        """
        self.spark = spark
        self.model_name = model_name

        self.pipeline_model = PretrainedPipeline(self.model_name, lang='en')

    @staticmethod
    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    def append_sentiment(pairs, sentiment) -> List[List[str]]:
        """Append sentiment to each entry in pred brand list. """
        return list(map(lambda x: x.append(sentiment), pairs))

    def predict(self, df: DataFrame) -> DataFrame:
        """Annotates the input dataframe with the classification results.

        Args:
          df : Pandas or Spark dataframe to classify (must contain a "text" column)
        """
        logging.info("Running sentiment model.")
        
        sentiment_df = self.pipeline_model.transform(df)
        w = Window.orderBy(F.monotonically_increasing_id())
        
        logging.info("Reorganising dataframe.")
        
        scores = sentiment_df \
            .select(F.explode(F.col("class.metadata")).alias("metadata")) \
            .select(F.col("metadata")["positive"].alias("positive"),
                    F.col("metadata")["neutral"].alias("neutral"),
                    F.col("metadata")["negative"].alias("negative")) \
            .withColumn("score", F.col("positive") - F.col("negative")) \
            .withColumn("column_index", F.row_number().over(w))
        
        logging.info("Calculating sentiment scores.")
        
        # Extract only target and label columns
        sentiment_df = sentiment_df.select(*self.SENTIMENT_FIELDS) \
            .withColumnRenamed("result", "Predicted_Sentiment") \
            .withColumn("Predicted_Sentiment",
                        F.array_join("Predicted_Sentiment", "")) \
            .withColumn("column_index", F.row_number().over(w))

        logging.info("Linking scores with dataframe.")

        # Join the predictions and the scores in one dataframe
        mask = sentiment_df.column_index == scores.column_index
        
        combined_df = sentiment_df \
            .join(scores, mask, "inner") \
            .drop(scores.column_index) \
            .drop(sentiment_df.column_index) \
            .withColumn('Predicted_Entity_and_Sentiment', 
                        self.append_sentiment('Predicted_Entity',
                                              'Predicted_Sentiment')) \
            .drop('Predicted_Entity', 'Predicted_Sentiment')

        return combined_df
