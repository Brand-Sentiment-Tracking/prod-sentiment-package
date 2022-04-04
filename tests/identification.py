import sparknlp
import unittest

from ..brand_sentiment.identification import BrandIdentification

# spark = sparknlp.start()
# article_extractor = ArticleExtraction()
# article = article_extractor.import_one_article('data/article.txt')
# print(article)
# sentences = article_extractor.article_to_sentences(article)
# print(sentences)

class TestBrandIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)

        self.spark = sparknlp.start()
        self.MODEL_NAME = ???
        self.brand = BrandIdentification(MODEL_NAME)
        self.text = ???
        

if __name__ == "__main__":
    unittest.main()