import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import pickle



def load_data(database_filepath):
    '''
    Reads in the table from the sqlite database as a dataframe and returns feature and response as two different dataframes
    
    Arguments:
        database_filepath -> path of the sqlite database
    
    Output:
        X -> dataframe containg the features
        y -> dataframe containg the response
        category_names -> list of the response columns
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql(database_filepath.split('/')[-1].replace(".db",""), con = engine)
    
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    category_names = y.columns
    
    return X,y,category_names


def tokenize(text):
    '''
    Normalize, remove punctuation, tokenize the words, remove stop words and lemmatize
    
    Arguments:
        text -> 
        
    Output:
        lemmed -> transformed text
    '''
    # Normalize the text
    clean_text = text.lower()
    # Remove punctuation
    clean_text = re.sub(r"[^a-zA-Z0-9]", " ", clean_text)
    # Tokenize the sentence
    words = word_tokenize(clean_text)
    # remove stop words
    words = [w for w in words if w not in stopwords.words('english')]
    # lemmatize or stem the words
    lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    
    return lemmed
    


def build_model():
    '''
    Instanciate the model
    
    Output:
        cv -> cross validation model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    
    # hyperparameters for the Random forrest classifier 
    parameters = {
        "clf__estimator__n_estimators": [50, 100, 200],
        "clf__estimator__criterion": ['gini', 'entropy'],
        "clf__estimator__max_depth": [2, 5]
    }
    
    # run cross validation
    cv = GridSearchCV(pipeline, param_grid = parameters, cv = 5)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function takes the CV model, selects the best performing model and prints f1_score, precision and recall for each category for the test dataset
    
    Arguments:
        model -> cross validation model
        X_test -> test messages
        Y_test -> true categories for the test messages
        category_names -> category names of the messages
        
    Output:
        
    '''
    best_model = model.best_estimator_
    Y_pred = best_model.predict(X_test)
    for i in range(Y_test.shape[1]):
        clf_result = classification_report(y_true = Y_test.values[:,i], y_pred = Y_pred[:,i], output_dict=True)['weighted avg']
        f1_score = clf_result['f1-score']
        precision = clf_result['precision']
        recall = clf_result['recall']
        print(f"For the CATEGORY {category_names[i]}, F1_SCORE is {f1_score} ,PRECISION is {precision} and RECALL is {recall}")
    


def save_model(model, model_filepath):
    '''
    This function takes the CV model, selects the best model and saves it as a pickle file
    
    Arguments:
        model -> CV model
        model_filepath -> path where model needs to be saved as a pickle file
    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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