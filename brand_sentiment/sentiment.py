from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, ArrayType
from sparknlp.pretrained import PretrainedPipeline


class SentimentIdentification:

    FIELDS = ("text", "source_domain", "date_publish", "language",
              "Predicted_Entity", "class.result")

    def __init__(self, spark: SparkSession, model_name: str):
        """Creates a class for sentiment identication using specified model.

        Args:
          model_name: Name of the Spark NLP pretrained pipeline.
        """
        self.spark = spark
        self.model_name = model_name

        self.pipeline_model = PretrainedPipeline(self.model_name, lang='en')

    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    @staticmethod
    def append_sentiment(pair_list, sentiment):
        """Append sentiment to each entry in pred brand list. """

        for pair in pair_list:
            pair.append(sentiment)

        return pair_list

    def predict_dataframe(self, df: DataFrame):
        """Annotates the input dataframe with the classification results.

        Args:
          df : Pandas or Spark dataframe to classify (must contain a "text" column)
        """
        sentiment_df = self.pipeline_model.transform(df)
        w = Window.orderBy(F.monotonically_increasing_id())

        scores = sentiment_df \
            .select(F.explode(F.col("class.metadata")).alias("metadata")) \
            .select(F.col("metadata")["positive"].alias("positive"),
                    F.col("metadata")["neutral"].alias("neutral"),
                    F.col("metadata")["negative"].alias("negative")) \
            .withColumn("score", F.col("positive") - F.col("negative")) \
            .withColumn("columnindex", F.row_number().over(w))

        # Extract only target and label columns
        sentiment_df = sentiment_df.select(*self.FIELDS) \
            .withColumnRenamed("result", "Predicted_Sentiment") \
            .withColumn("Predicted_Sentiment",
                        F.array_join("Predicted_Sentiment", "")) \
            .withColumn("columnindex", F.row_number().over(w))

        # Join the predictions and the scores in one dataframe
        mask = sentiment_df.columnindex == scores.columnindex
        combined_df = sentiment_df \
            .join(scores, mask, "inner") \
            .drop(scores.columnindex) \
            .drop(sentiment_df.columnindex) \
            .withColumn('Predicted_Entity_and_Sentiment', 
                        self.append_sentiment('Predicted_Entity',
                                              'Predicted_Sentiment')) \
            .drop('Predicted_Entity', 'Predicted_Sentiment')

        return combined_df
