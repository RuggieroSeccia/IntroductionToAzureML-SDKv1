import os
from pathlib import Path

import click
import pandas as pd

from DataTransformer import DataTransformer


@click.command()
@click.option('--input_dir', type=Path, help='Path of the input data')
@click.option('--output_dir', type=Path, help='Path of the output data')
def main(input_dir, output_dir):
    """Call the DataTransformer in Azure
    Saves the Transformed dataframe as a pickle in the directory output_dir/output_file

    :param input_dir: path of the input file
    :param output_dir: path of the output file
    """
    print('********** DataTransformer_AML **********')

    print('Retrieving the input data...')
    df = pd.read_pickle(os.path.join(input_dir, 'df.pickle'))
    print('Data retrieved correctly')

    print('Calling the DataTransformer...')
    df_transformed = DataTransformer.execute(df)
    print('DataTransformer executed')

    print('Saving the transformed data....')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, 'df.pickle')
    df_transformed.to_pickle(output_path)
    print(f'Transformed data saved at location {output_path}')


if __name__ == "__main__":
    main()
