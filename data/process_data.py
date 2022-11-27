# import necessary packages

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the two CSVs and joins them on id to generate a combined dataset
    
    Arguments:
        messages_filepath -> path to messages.csv file
        categories_filepath -> path to categories.csv file
        
    Output:
        df -> dataframe combining categories and messages data
    '''
    # reads in the messages csv file
    messages = pd.read_csv(messages_filepath)
    
    # reads in the categories csv file
    categories = pd.read_csv(categories_filepath)
    
    # merge the two dataframes
    df = pd.merge(messages, categories, how='inner', on=["id"])
    
    #return the dataframe
    return df


def clean_data(df):
    '''
    Cleans the combined dataframe that is passed in
    
    Arguments:
        df -> combined dataframe of messages and categories
    
    Output:
        df -> cleaned dataframe
    '''
    # create a new dataframe from the categories column with each ; seperated element as new column
    categories = df["categories"].str.split(";", expand = True)
    row = categories.iloc[0,:]
    categories_column = row.apply(lambda x : x.split("-")[0]).to_list()
    categories.columns = categories_column
    
    # get the 0 and 1 from the columns of categories dataframe
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype('int64')
        
    df.drop(columns = ["categories"], inplace=True)
    # join the two dataframes on the index
    df = df.join(categories)
    
    # drop the duplicate rows if exists
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Gets the dataframe and database name and saves the dataframe as table in database
    
    Arguments:
        df -> dataframe that needs to be saved as table in the database
        database_filename -> name of the sqlite database
    '''
    # define the connection to the database
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename.replace(".db", ""), engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()