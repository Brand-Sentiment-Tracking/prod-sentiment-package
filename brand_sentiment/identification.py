import logging

from typing import List, Tuple

from pyspark import Row
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import functions as F

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import XlnetForTokenClassification, \
    Tokenizer, BertEmbeddings, NerDLModel, NerConverter
from transformers import PreTrainedModel


class BrandIdentification:

    NER_FIELDS = ("text", "source_domain", "date_publish",
                  "language", "entities")

    def __init__(self, spark: SparkSession, model_name: str,
                 partitions: int = 32, log_level: int = logging.INFO):

        self.logger = logging.getLogger("BrandIdentification")
        self.logger.setLevel(log_level)

        self.spark = spark

        self.model_name = model_name
        self.partitions = partitions

    @property
    def model(self) -> PreTrainedModel:
        return self.__model

    @property
    def pipeline(self) -> Pipeline:
        return self.__pipeline

    @property
    def model_name(self) -> str:
        return self.__model_name

    @model_name.setter
    def model_name(self, name: str):
        if name not in ("xlnet_base", "ner_conll_bert_base_cased"):
            raise ValueError("Model must be either 'xlnet_base' or "
                             "'ner_conll_bert_base_cased'.")

        self.__model_name = name
        self.__build_pipeline()

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

    def __build_document_stages(self) -> List:
        document_assembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        return [document_assembler, tokenizer]

    def __build_converter_stages(self) -> List:
        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')
        
        return [ner_converter]

    def __build_xlnet_model_stages(self) -> List:
        token_classifier = XlnetForTokenClassification \
            .pretrained('xlnet_base_token_classifier_conll03', 'en') \
            .setInputCols(['token', 'document']) \
            .setOutputCol('ner') \
            .setCaseSensitive(True) \
            .setMaxSentenceLength(512)

        return [token_classifier]

    def __build_conll_model_stages(self) -> List:
        embeddings = BertEmbeddings \
            .pretrained(name='bert_base_cased', lang='en') \
            .setInputCols(['document', 'token']) \
            .setOutputCol('embeddings')

        ner_model = NerDLModel \
            .pretrained(self.model_name, 'en') \
            .setInputCols(['document', 'token', 'embeddings']) \
            .setOutputCol('ner')

        return [embeddings, ner_model]

    def __build_pipeline(self):
        self.logger.info("Building NER Pipeline...")
        self.logger.info("Building Document Assembler & Tokeniser.")
        stages = self.__build_document_stages()

        if self.model_name == "xlnet_base":
            self.logger.info("Building XLNet Model.")
            stages.extend(self.__build_xlnet_model_stages())

        elif self.model_name == "ner_conll_bert_base_cased":
            self.logger.info("Building CoNLL BERT Model.")
            stages.extend(self.__build_conll_model_stages())
        else:
            logging.fatal("No matching model name for pipeline. Should have "
                          "thrown a ValueError when setting the model name.")

        self.logger.info("Build NER Converter.")
        stages.extend(self.__build_converter_stages())

        # An empty df with column name "text"
        empty_df = self.spark.createDataFrame([['']], ["text"])

        self.__pipeline = Pipeline(stages=stages)
        self.__model = self.pipeline.fit(empty_df)

        self.logger.info("NER Pipeline built successfully.")

    @staticmethod
    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    def __extract_brands(rows: List[Row]) -> List[Tuple[str, str]]:
        return [[row.result, row.metadata['entity']] for row in rows]

    def predict_brand(self, df: DataFrame,
                      filter_no_entities: bool = False) -> DataFrame:

        self.logger.info("Running NER model.")

        brand_df = self.model.transform(df) \
            .withColumn("entities", self.__extract_brands('ner_chunk')) \
            .select(*self.NER_FIELDS)

        return self.remove_articles_without_entities(brand_df) \
            if filter_no_entities else brand_df.repartition(self.partitions)

    def remove_articles_without_entities(self, df: DataFrame) -> DataFrame:
        self.logger.info("Removing articles with no entities.")

        return df.filter(F.size(df.entities) > 0) \
            .repartition(self.partitions)
