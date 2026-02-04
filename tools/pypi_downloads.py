import logging
import os.path

import click
import pandas as pd
from hbutils.logging import ColoredFormatter
from huggingface_hub import hf_hub_download


@click.command()
@click.option('--dst-csv-file', type=click.Path(), default='pypi_downloads.csv',
              help='Destination CSV file path to save the filtered data.', show_default=True)
@click.option('--min-last-month', type=int, default=500000,
              help='Minimum number of downloads in the last month to filter packages.', show_default=True)
def sync(dst_csv_file: str, min_last_month: int = 500000):
    df = pd.read_parquet(hf_hub_download(
        repo_id='HansBug/pypi_downloads',
        repo_type='dataset',
        filename='dataset.parquet',
    ))
    df = df[df['status'] == 'valid']
    df[['last_day', 'last_week', 'last_month']] = df[['last_day', 'last_week', 'last_month']].astype(int)
    logging.info(f'Original Full Table:\n{df}')

    df = df.sort_values(by=['last_month'], ascending=[False])
    df = df.drop(columns=['status'])
    df = df[df['last_month'] >= min_last_month]
    logging.info(f'Filtered Table:\n{df}')

    logging.info(f'Saving to {dst_csv_file!r} ...')
    if os.path.dirname(dst_csv_file):
        os.makedirs(os.path.dirname(dst_csv_file), exist_ok=True)
    df.to_csv(dst_csv_file, index=False)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    sync()
