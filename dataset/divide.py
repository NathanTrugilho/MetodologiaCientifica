import pandas as pd
from sklearn.model_selection import train_test_split

# Carrega o dataset já processado
df = pd.read_csv('dataset_differences.csv')

# Shuffle e divisão em treino (70%) + temp (30%)
train_df, temp_df = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)

# Divide temp em validação (10%) e teste (20%)
# 10% de total = 10/30 ≈ 0.333 da parte temporária
val_df, test_df = train_test_split(temp_df, test_size=0.666, shuffle=True, random_state=42)

# Salva os arquivos
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Dados divididos e salvos como: train.csv, validation.csv e test.csv")
