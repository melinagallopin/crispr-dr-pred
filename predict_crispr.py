from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib
import sys


def read_dataset(filename):
    df = pd.read_csv(filename)

    # move some variables to the end of the table
    move_to_end(df, "rnafold_37_dot_par")
    move_to_end(df, "rnafold_75_dot_par")
    move_to_end(df, "blastscore")
    move_to_end(df, 'cas_prox_class')

    # Construct cas_prox_subtype and cas_prox_type  target variables
    df = df.rename(columns={'cas_prox_class': 'cas_prox_subtype'})
    cas_prox_subtype = list(map(lambda s: s.replace('IV', 'IIII'), df.cas_prox_subtype))
    cas_prox_subtype = list(map(lambda s: s.replace('V', 'IIIII'), cas_prox_subtype))
    cas_prox_subtype = list(map(lambda s: s.replace('VI', 'IIIIII'), cas_prox_subtype))
    df['cas_prox_type'] = list(map(lambda s: s.count('I'), cas_prox_subtype))

    # Construct crispr/not-crispr target variable
    df = df.astype({"blastscore": float})
    df = df.assign(crispr=np.repeat(-1, df.shape[0]))
    # all el4 are good
    df.crispr[df.evidencelevel == 4] = 1
    # all el1 are not good
    df.crispr[df.evidencelevel == 1] = 0
    # all el1 that have blast score higher than 40 are reassigned to el4
    df.crispr[np.logical_and(df.evidencelevel == 1, df.blastscore > 40)] = 1
    # we exclude from the 0 dataset all DR than have blast < 40 but >10
    df.crispr[np.logical_and.reduce((df.evidencelevel == 1, df.blastscore < 40, df.blastscore > 10))] = -1

    # canonical k-mers
    df2 = remove_repetitive_id(df)
    to_keep = find_sufficient_mers(df.columns[11:75]) + find_sufficient_mers(df.columns[75:331])
    to_delete = pd.Index(np.setdiff1d(df.columns[11:331], to_keep))
    df2 = df2.drop(to_delete, axis='columns')
    return df2


def move_to_end(df, col_name):
    col = df[col_name]
    df.pop(col_name)
    idx = df.shape[1]
    df.insert(idx, col_name, col)


def move_to_begining(df, col_name):
    col = df[col_name]
    df.pop(col_name)
    df.insert(0, col_name, col)


def remove_repetitive_id(df):
    """
    Keep those with a minimal rnafold 37 energy
    """

    def _for_one_unique_id(id_):
        repetitive_instances = df.loc[df.id == id_]
        repetitive_ids = repetitive_instances.index
        min_energy_id = repetitive_ids[df.rnafold_37_energy[repetitive_ids].argmin()]
        final_example = df.loc[min_energy_id, :]
        final_example.iloc[11:331] = repetitive_instances.iloc[:, 11:331].sum(axis=0)
        return final_example

    many_unique_ids = np.vectorize(_for_one_unique_id)
    unique_ids = np.unique(df.id)
    unique_rows = many_unique_ids(unique_ids)
    return pd.concat(unique_rows, axis=1).T


def reverse_complement(seq):
    rc = seq[::-1]
    rc = rc.replace('A', '*')
    rc = rc.replace('T', 'A')
    rc = rc.replace('*', 'T')
    rc = rc.replace('C', '*')
    rc = rc.replace('G', 'C')
    rc = rc.replace('*', 'G')
    return rc


def find_sufficient_mers(mers):
    res = []
    for mer in mers:
        rc = reverse_complement(mer)
        if rc == mer:
            res.append(mer)
        else:
            if rc not in res:
                res.append(mer)
    return res


if __name__ == '__main__':
    fname = sys.argv[0]
    data = read_dataset(fname)
    fourmers = joblib.load('models/crispr/4mers-model.pkl')
    y_pred = fourmers.predict(data.iloc[:, 43:179])
    print(accuracy_score(data.crispr, y_pred))
