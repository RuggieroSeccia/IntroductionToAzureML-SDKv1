from pathlib import Path
import os

import click

from DataFetcher import DataFetcher


@click.command()
@click.option('--input_dataset_name', type=str, help='Name of the registered dataset to load')
@click.option("--input_dataset_version", type=str, default="latest",
              help="version number of the registered dataset")
@click.option('--output_dir', type=Path, help='Path to the output dataframe')
def main(input_dataset_name, input_dataset_version, output_dir):
    """Fetches the data in Azure

    :param input_dataset_name: name of the registered dataset to load
    :param input_dataset_version: version of the registered dataset to load
    :param output_dir: directory of the output file
    :return: None
    """

    print('********** DataFetcher_AML **********')
    df = DataFetcher.execute(input_dataset_version=input_dataset_version,
                             input_dataset_name=input_dataset_name,
                             )
    print('Saving the data...')
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_pickle(os.path.join(output_dir, 'df.pickle'))
    print(f'Dataframe saved as pickle at location {os.path.join(output_dir, "df.pickle")}')


if __name__ == "__main__":
    main()
