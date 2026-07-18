import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def run_preprocessing(input_path, output_dir):
    df = pd.read_csv(input_path)

    # Melt and Clean
    prod_cols = [col for col in df.columns if 'Produksi' in col and '(kuintal)' in col]
    df_melted = df.melt(id_vars=['Provinsi'], value_vars=prod_cols, var_name='Komoditas', value_name='Prod_Kw')
    df_melted['Prod_Kw'] = df_melted['Prod_Kw'].replace('-', np.nan).astype(float)
    df_melted = df_melted.dropna().copy()
    df_melted['produksi_ton'] = df_melted['Prod_Kw'] / 10

    # Encoding
    le = LabelEncoder()
    df_melted['prov_enc'] = le.fit_transform(df_melted['Provinsi'])
    df_melted['komo_enc'] = le.fit_transform(df_melted['Komoditas'])

    # Save
    os.makedirs(output_dir, exist_ok=True)
    df_melted.to_csv(os.path.join(output_dir, 'automated_preprocessed.csv'), index=False)
    print("Preprocessing otomatis selesai.")

if __name__ == '__main__':
    run_preprocessing('../Produksi_Tanaman_Raw.csv', 'namadataset_preprocessing')
