import pandas as pd
import argparse
import os 


def get_users(dataframe, col_names):
    users = set()

    for row in dataframe[col_names].iterrows():
        for user in row[1]:
            if type(user) == str and '|' in user:
                id = user.split('|')[-1]
                users.add(id)
    
    return users

def write_file(users):
    with open('users.txt', 'w') as f:
        for user in users:
            f.write(user + '\n')
            
parser = argparse.ArgumentParser(description='Fetch tweets.')
parser.add_argument('--data_dir', help='data directory')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir

    sarcastic_path = os.path.join(data_dir, 'SPIRS-sarcastic.csv')
    nonsarcastic_path = os.path.join(data_dir, 'SPIRS-non-sarcastic.csv')

    sarcastic_df = pd.read_csv(sarcastic_path, sep=None)
    sarcastic_df['label'] = [1] * len(sarcastic_df)

    nonsarcastic_df = pd.read_csv(nonsarcastic_path, sep=None)
    nonsarcastic_df['label'] = [0] * len(nonsarcastic_df)

    col_names = ['cue_user', 'sar_user', 'obl_user', 'eli_user']
    users = get_users(sarcastic_df, col_names)
    users.union(get_users(nonsarcastic_df, col_names[1:]))
    write_file(users)