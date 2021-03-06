import json
import plotly
import pandas as pd

from nltk import sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    A class to get the length of each tokenized text, and apply the function to
    all cells
    '''
    def textlength(self, text):
        '''
        set a function to get the length of the tokenized text
        '''
        return len(tokenize(text))

    def fit(self, x, y=None):
        '''
        set a fit function
        '''
        return self

    def transform(self, X):
        '''
        set a function to apply the textlength to all texts using a
        pd.Series.apply
        '''
        X_tagged = pd.Series(X).apply(self.textlength)
        return pd.DataFrame(X_tagged)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    A class to see if the first letter is a verb, and apply the function to all
    cells
    '''
    def starting_verb(self, text):
        '''
        set a function to judge if the first token of the text is a verb
        '''
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        '''
        set a fit function
        '''
        return self

    def transform(self, X):
        '''
        set a function to apply the starting_verb to all texts using a
        pd.Series.apply
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("../models/model_randomforest.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''
    set a function to render the master.html
    '''
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    basic_classification_counts = df.iloc[:,4:7].sum().\
                                        sort_values(ascending = False)
    basic_classification_names = list(basic_classification_counts.index)

    indicator_counts = df.iloc[:,7:].sum().sort_values(ascending = False)
    indicator_names = list(indicator_counts.index.str.replace('_', ' '))

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x = basic_classification_names,
                    y = basic_classification_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Basic Classification',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },

        {
            'data': [
                Bar(
                    x = indicator_names,
                    y = indicator_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Indicator Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    '''
    set a function to render the go.html
    '''
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    '''
    set the main function
    '''
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
