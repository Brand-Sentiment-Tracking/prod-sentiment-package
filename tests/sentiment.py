import sparknlp
import unittest

from ..brand_sentiment.sentiment import SentimentIdentification

# spark = sparknlp.start()
# article_extractor = ArticleExtraction()
# article = article_extractor.import_one_article('data/article.txt')
# print(article)
# sentences = article_extractor.article_to_sentences(article)
# print(sentences)

class TestSentimentIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        """ Include useful things for all test cases - keep minimal
        """
        super().__init__(*args, **kwargs)

        self.spark = sparknlp.start()
        self.identifier = SentimentIdentification()


if __name__ == "__main__":
    unittest.main()