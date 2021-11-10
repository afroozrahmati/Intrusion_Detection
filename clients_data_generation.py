import os
import pandas as pd

df_normal = pd.read_csv('data/normal.csv')
df_abnormal = pd.read_csv('data/abnormal.csv')
number_clinets=10

normal_data_rows_per_client = len(df_normal)//number_clinets
abnormal_data_rows_per_client = len(df_abnormal)//number_clinets

def split(filehandler, delimiter=',', row_limit=1000,
          output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)

split(open('data/normal.csv', 'r'),row_limit=normal_data_rows_per_client);
split(open('data/abnormal.csv', 'r'),row_limit=abnormal_data_rows_per_client);