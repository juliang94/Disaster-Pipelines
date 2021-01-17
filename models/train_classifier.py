import sys

import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

import pickle



def load_data(database_filepath):
    """
    This fuction loads the cleaned data from the database
    Outputs are the features, target features, and category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    
    X = df['message']
    Y = df.iloc[:, 4:]
    categories = list(Y.columns)

    return X, Y, categories


def tokenize(text):
    """
    This function preprocesses the message data and returns the tokens for each one.
    """
    ## tokenize text
    tokens = word_tokenize(text)
    
    ## lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        
        
        ## remove punctiation
        clean_token = re.sub(r'[^a-zA-Z0-9]', '', token)
        
        ## lemmatize and append to list
        clean_token = lemmatizer.lemmatize(clean_token).lower()
        
        clean_tokens.append(clean_token)
        
    clean_tokens = [x for x in clean_tokens if x]
 
    
    return clean_tokens


def build_model():
    """
    This function uses a machine learning pipeline and trains an AdaBoost classifier on the tokenized messages
    """
    ## classifiaction model
    clf = AdaBoostClassifier()
    
    ## construct pipeline with tokenizer, NLP transformers, and classifier
    model = Pipeline([
    ('features', FeatureUnion([
        ('text pipeline', Pipeline([
            ('counting', CountVectorizer(tokenizer= tokenize)),
            ('tfidf', TfidfTransformer())
        ]))
    ])),
    ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.2, 0.3],
        'clf__estimator__n_estimators': [50, 200, 300]
    }

    cv_model = GridSearchCV(model, param_grid= parameters, verbose = 3)
    
    return cv_model


def evaluate_model(model, X_test, Y_test, category_names):

    """
    The optimal model is evaluated here by making predictions and printing out a classifier report
    """
    ## print best parameters
    best_params = model.best_params_
    print(best_params)
    
    ## predictions
    y_preds = model.predict(X_test)

    ## class report
    class_report = classification_report(Y_test, y_preds, target_names = category_names)
    print(class_report)
    
    return model


def save_model(model, model_filepath):
    """
    The optimal model is saved as a pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        

        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
