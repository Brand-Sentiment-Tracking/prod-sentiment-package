import logging

from typing import List, Tuple
from pandas import DataFrame

from pyspark import Row
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, ArrayType

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import XlnetForTokenClassification, \
    Tokenizer, BertEmbeddings, NerDLModel, NerConverter


class BrandIdentification:

    NER_FIELDS = ("text", "source_domain", "date_publish",
                  "language", "Predicted_Entity")

    def __init__(self, spark: SparkSession, model_name: str):
        self.spark = spark
        self.model_name = model_name

        self.model = None
        self.build_pipeline()

    def build_pipeline(self):
        logging.info("Building NER Pipeline...")
        
        document_assembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')


        if self.model_name == "xlnet_base":
            tokenClassifier = XlnetForTokenClassification \
                .pretrained('xlnet_base_token_classifier_conll03', 'en') \
                .setInputCols(['token', 'document']) \
                .setOutputCol('ner') \
                .setCaseSensitive(True) \
                .setMaxSentenceLength(512)

            nlp_pipeline = Pipeline(stages=[
                document_assembler, 
                tokenizer,
                tokenClassifier,
                ner_converter
            ])

        elif self.model_name == "ner_conll_bert_base_cased":
            # Bert model uses Bert embeddings
            embeddings = BertEmbeddings \
                .pretrained(name='bert_base_cased', lang='en') \
                .setInputCols(['document', 'token']) \
                .setOutputCol('embeddings')

            ner_model = NerDLModel \
                .pretrained(self.model_name, 'en') \
                .setInputCols(['document', 'token', 'embeddings']) \
                .setOutputCol('ner')

            nlp_pipeline = Pipeline(stages=[
                document_assembler, 
                tokenizer,
                embeddings,
                ner_model,
                ner_converter
            ])

        else:
            raise ValueError("Model must be either 'xlnet_base' or "
                             "'ner_conll_bert_base_cased'.")

        # An empty df with column name "text"
        empty_df = self.spark.createDataFrame([['']], ["text"])
        self.model = nlp_pipeline.fit(empty_df)
        
        logging.info("NER Pipeline built successfully.")

    @staticmethod
    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    def extract_brands(rows: List[Row]) -> List[Tuple[str, str]]:
        return [[row.result, row.metadata['entity']] for row in rows]

    def predict(self, df: DataFrame) -> DataFrame:
        logging.info("Running NER model.")
        
        brand_df = self.model.transform(df) \
            .withColumn("Predicted_Entity", self.extract_brands('ner_chunk')) \
            .select(*self.NER_FIELDS)
        
        logging.info("Removing articles with no entities.")
        return brand_df.filter(F.size(brand_df.Predicted_Entity) > 0)
