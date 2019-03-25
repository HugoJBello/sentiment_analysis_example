# https://www.pingshiuanchua.com/blog/post/simple-sentiment-analysis-python?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com


import nltk
nltk.download('vader_lexicon')
import pandas as pd

def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score


dataset = ["Food is good and not too expensive. Serving is just right for adult. Ambient is nice too.",
           "Used to be good. Chicken soup was below average, bbq used to be good.",
           "Food was good, standouts were the spicy beef soup and seafood pancake! ",
           "Good office lunch or after work place to go to with a big group as they have a lot of private areas with large tables",
           "As a Korean person, it was very disappointing food quality and very pricey for what you get. I won't go back there again. ",
           "Not great quality food for the price. Average food at premium prices really.",
           "Fast service. Prices are reasonable and food is decent.",
           "Extremely long waiting time. Food is decent but definitely not worth the wait.",
           "muy muy bueno me ha molado mucho. Es genial. Me encanta",
           "no me ha gustado nada. No pienso volver. Es horrible. Lo odio."
           "Clean premises, tasty food. My family favourites are the clear Tom yum soup, stuffed chicken wings, chargrilled squid.",
           "really good and authentic Thai food! in particular, we loved their tom yup clear soup with sliced fish. it's so well balanced that we can have it just on its own. "
           ]

nltk_results = [nltk_sentiment(row) for row in dataset]
results_df = pd.DataFrame(nltk_results)
text_df = pd.DataFrame(dataset, columns = ['text'])
nltk_df = text_df.join(results_df)
print(nltk_df);