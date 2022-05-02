import logging

from typing import List

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.window import Window, WindowSpec
from pyspark.sql.types import StringType, ArrayType
from sparknlp.pretrained import PretrainedPipeline


class SentimentIdentification:

    SENTIMENT_FIELDS = ("text", "source_domain", "date_publish", "language",
                        "entities", "class.result")

    def __init__(self, spark: SparkSession, model_name: str,
                 partitions: int = 32, log_level: int = logging.INFO):
        """Creates a class for sentiment identication using specified model.

        Args:
          model_name: Name of the Spark NLP pretrained pipeline.
        """
        self.logger = logging.getLogger("SentimentIdentification")
        self.logger.setLevel(log_level)

        self.spark = spark
        self.model_name = model_name

        self.partitions = partitions

    @property
    def model(self) -> PretrainedPipeline:
        return self.__model

    @property
    def model_name(self) -> str:
        return self.__model_name

    @model_name.setter
    def model_name(self, name: str):
        if type(name) != str:
            raise TypeError("Model name is not a string.")
        if name != "classifierdl_bertwiki_finance_sentiment_pipeline":
            self.logger.warning("Pipeline hasn't been designed for model "
                                f"'{name}'. Using this model may cause the "
                                "pipeline to crash.")

        self.__model_name = name
        self.__model = PretrainedPipeline(self.model_name, lang='en')

    @property
    def partitions(self) -> int:
        return self.__partitions

    @partitions.setter
    def partitions(self, n: int):
        if type(n) != int:
            raise ValueError("Partitions is not an integer")
        elif n <= 0:
            raise ValueError("Partitions is not greater than 0.")

        self.__partitions = n

    def __get_sentiment_scores(self, df: DataFrame,
                               window: WindowSpec) -> DataFrame:
        return df \
            .select(F.explode(F.col("class.metadata")).alias("metadata")) \
            .select(F.col("metadata")["positive"].alias("positive"),
                    F.col("metadata")["neutral"].alias("neutral"),
                    F.col("metadata")["negative"].alias("negative")) \
            .withColumn("score", F.col("positive") - F.col("negative")) \
            .withColumn("column_index", F.row_number().over(window))

    def __reorganise_df(self, df: DataFrame, window: WindowSpec) -> DataFrame:
        return df \
            .select(*self.SENTIMENT_FIELDS) \
            .withColumnRenamed("result", "sentiment") \
            .withColumn("sentiment", F.array_join("sentiment", "")) \
            .withColumn("column_index", F.row_number().over(window))

    @staticmethod
    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    def __append_sentiment(entities, sentiment) -> List[List[str]]:
        """Append sentiment to each entry in pred brand list. """
        for entity in entities:
            entity.append(sentiment)

        return entities

    def __add_scores(self, df: DataFrame, scores: DataFrame) -> DataFrame:
        mask = df.column_index == scores.column_index

        return df.join(scores, mask, "inner") \
            .drop(scores.column_index) \
            .drop(df.column_index) \
            .withColumn('Predicted_Entity_and_Sentiment',
                        self.__append_sentiment('entities', 'sentiment')) \
            .drop('entities', 'sentiment')

    def predict_sentiment(self, brand_df: DataFrame) -> DataFrame:
        """Annotates the input dataframe with the classification results.

        Args:
            df : Pandas or Spark dataframe to classify (must contain
                a "text" column)
        """
        self.logger.info("Running sentiment model...")
        df = self.model.transform(brand_df)

        w = Window.orderBy(F.monotonically_increasing_id())

        self.logger.info("Calculating sentiment scores.")
        scores = self.__get_sentiment_scores(df, w)

        self.logger.info("Reorganising dataframe.")
        df = self.__reorganise_df(df, w)

        self.logger.info("Adding scores to dataframe.")
        
        return self.__add_scores(df, scores) \
            .repartition(self.partitions)
