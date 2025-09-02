import pandas as pd

# Carrega o Excel (troque o nome do arquivo)
df = pd.read_excel('dataset.xlsx')

# Arredondar as idades
df['winner_age'] = df['winner_age'].round().astype(int)
df['loser_age'] = df['loser_age'].round().astype(int)

# Codificar 'hand' como numérico (exemplo: 'R' = 1, 'L' = 0)
hand_map = {'R': 1, 'L': 0}
df['winner_hand_num'] = df['winner_hand'].map(hand_map)
df['loser_hand_num'] = df['loser_hand'].map(hand_map)

# Criar variável única para o chão: 1 se 'hard', 0 se 'clay' ou outros
df['surface_hard'] = df['surface'].apply(lambda x: 1 if x.lower() == 'hard' else 0)

# Função para criar um dataframe com as diferenças e target
def criar_dataset_diff(df):
    df_diff = pd.DataFrame()
    df_diff['diff_ht'] = df['winner_ht'] - df['loser_ht']
    df_diff['diff_age'] = df['winner_age'] - df['loser_age']
    df_diff['diff_rank'] = df['winner_rank'] - df['loser_rank']
    df_diff['diff_hand'] = df['winner_hand_num'] - df['loser_hand_num']
    df_diff['surface_hard'] = df['surface_hard']  # mesma superfície pra ambos

    df_diff['target'] = 1

    # Dataset invertido
    df_diff_inv = pd.DataFrame()
    df_diff_inv['diff_ht'] = df['loser_ht'] - df['winner_ht']
    df_diff_inv['diff_age'] = df['loser_age'] - df['winner_age']
    df_diff_inv['diff_rank'] = df['loser_rank'] - df['winner_rank']
    df_diff_inv['diff_hand'] = df['loser_hand_num'] - df['winner_hand_num']
    df_diff_inv['surface_hard'] = df['surface_hard']

    df_diff_inv['target'] = 0

    df_final = pd.concat([df_diff, df_diff_inv], ignore_index=True)
    return df_final

df_dataset = criar_dataset_diff(df)

df_dataset.to_csv('dataset_differences.csv', index=False)

print("Dataset criado e salvo com sucesso!")
