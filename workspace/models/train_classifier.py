
import sys
from sqlalchemy import create_engine
import pandas as pd
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    '''
    Load dataframe from a database
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM message', engine)
    X = df.message
    y = df.iloc[:, 4:]
    category_names = list(y.columns)
    
    return X, y, category_names

def tokenize(text):
    '''
    Tokenize and lemmatize the text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    A class to get the length of each tokenized text, and apply the function to all cells
    '''
    def textlength(self, text):        
        return len(tokenize(text))

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.textlength)
        return pd.DataFrame(X_tagged)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    A class to see if the first letter is a verb, and apply the function to all cells
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Build the model
    '''
    pipeline_randomforest = Pipeline([
    ('features', FeatureUnion([

        ('nlp_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize,
                                     max_df = 0.5,
                                     max_features = 5000)),
            ('tfidf', TfidfTransformer(use_idf = True))
             ])),

        ('txt_len', TextLengthExtractor()),
        ('start_verb', StartingVerbExtractor())
    ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_leaf =1,
                                                         n_estimators = 1000,
                                                         min_samples_split = 4)))
    ])
    
    return pipeline_randomforest


def evaluate_model(model, X_test, y_test, category_names):
    '''
    use the model to make prediction, and print out every column's precision, recall and fi scores
    '''
    y_pred = model.predict(X_test)
    
    result = []
    for i in range(y_test.shape[1]): 
        test_value = y_test.iloc[:, i]
        pred_value = [a[i] for a in y_pred]
        result.append(list(classification_report(test_value, pred_value,output_dict = True)['0'].values())[:3])
    
    df = pd.DataFrame(result,columns=['precision','recall','f1_score'])
    df['indicator'] = pd.Series(category_names)
    
    print(df)
    print('The average precision, recall and f1_score are {},{},{}'.
          format(df.precision.mean(),df.recall.mean(),df.f1_score.mean()))

def save_model(model, model_filepath):
    '''
    Save the model to a .pkl file
    '''
    pickle.dump(model, open('model_randomforest.pkl', 'wb'))


def main():
    
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
    
