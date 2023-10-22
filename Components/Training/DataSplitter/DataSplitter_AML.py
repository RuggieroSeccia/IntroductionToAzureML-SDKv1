import os
from pathlib import Path

import click
import pandas as pd

from DataSplitter import DataSplitter


@click.command()
@click.option('--input_dir', type=Path, help='Path to the input data')
@click.option('--output_train_data_dir', type=Path, help='Path where the training data are stored')
@click.option('--output_test_data_dir', type=Path, help='Path where the test data are stored')
@click.option('--target_feat', type=str, help='Name of the column in the DataFrame containing the output')
@click.option('--test_size', default=0.25, help='Percentage of samples left out for testing')
@click.option('--random_state', type=int, default=1, help='Integer to set the random seed')
def main(input_dir, output_train_data_dir, output_test_data_dir,
         target_feat, test_size, random_state):
    """Call the DataSplitter in Azure

    Saves the split data as pickle files

    :param input_dir: directory of the input data
    :param output_train_data_dir: directory where the training data are stored
    :param output_test_data_dir: directory where the test data are stored
    :param target_feat: name of the column in the DataFrame containing the output
    :param test_size: percentage of samples left out for testing
    :param random_state: integer to set the random seed
    :return: None
    """

    print('********** DataSplitter_AML **********')
    print('Loading the data....')
    df = pd.read_pickle(os.path.join(input_dir, 'df.pickle'))
    print('Data loaded')

    print('Calling the DataSplitter...')
    x_train, x_test, y_train, y_test = DataSplitter.execute(df=df,
                                                            target_feat=target_feat,
                                                            test_size=test_size,
                                                            random_state=random_state)
    print('DataSplitter executed')

    print('Saving the split dataset...')
    output_train_data_dir.mkdir(parents=True, exist_ok=True)
    x_train.to_pickle(os.path.join(output_train_data_dir, 'x_train.pickle'))
    y_train.to_pickle(os.path.join(output_train_data_dir, 'y_train.pickle'))
    print('Training data saved')

    output_test_data_dir.mkdir(parents=True, exist_ok=True)
    x_test.to_pickle(os.path.join(output_test_data_dir, 'x_test.pickle'))
    y_test.to_pickle(os.path.join(output_test_data_dir, 'y_test.pickle'))
    print('Test data saved')


if __name__ == "__main__":
    main()
