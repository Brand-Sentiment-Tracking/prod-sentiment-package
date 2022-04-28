from typing import List, Tuple
from pandas import DataFrame

from pyspark import Row
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, ArrayType

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, BertEmbeddings, \
    NerDLModel, NerConverter


class BrandIdentification:

    FIELDS = ("text", "source_domain", "date_publish",
              "language", "Predicted_Entity")

    def __init__(self, spark: SparkSession, model_name: str):
        self.spark = spark
        self.model_name = model_name

        self.model = None
        self.build_pipeline()

    def build_pipeline(self):
        document_assembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        embeddings = BertEmbeddings \
            .pretrained(name='bert_base_cased', lang='en') \
            .setInputCols(['document', 'token']) \
            .setOutputCol('embeddings')

        ner_model = NerDLModel \
            .pretrained(self.model_name, 'en') \
            .setInputCols(['document', 'token', 'embeddings']) \
            .setOutputCol('ner')

        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')

        nlp_pipeline = Pipeline(stages=[
            document_assembler,
            tokenizer,
            embeddings,
            ner_model,
            ner_converter
        ])

        # An empty df with column name "text"
        empty_df = self.spark.createDataFrame([['']], ["text"])
        self.model = nlp_pipeline.fit(empty_df)

    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    @staticmethod
    def extract_brands(rows: List[Row]) -> List[Tuple[str, str]]:
        return [(row.result, row.metadata['entity']) for row in rows]

    def predict_brand(self, df: DataFrame) -> DataFrame:
        brand_df = self.model.transform(df) \
            .withColumn("Predicted_Entity", self.extract_brands('ner_chunk')) \
            .select(*self.FIELDS)

        return brand_df.filter(F.size(brand_df.Predicted_Entity) > 0)
