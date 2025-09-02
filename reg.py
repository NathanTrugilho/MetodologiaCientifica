from pysr import PySRRegressor
import numpy as np

ARQUIVO_CONTROLE = "horas_treino.txt"

# Carrego os dados da planilha para a memória ==================
dados_planilha = np.loadtxt("dados/treino.csv", delimiter = ',' , dtype = float, skiprows=1)

# Faço a divisão entre as variáveis dependentes e independentes 
var_independentes = dados_planilha[ : ,  [0, 1, 2, 3, 4]]  # Contém somente X                        
var_dependentes = dados_planilha[ : , 5] # Contém somente Y

# Definindo meu modelo ===================
modelo = PySRRegressor(
    model_selection="best",
    niterations=100000,  # Botei um número bem alto para a condição de parada ser dada apenas pelo tempo
    populations=100, # Default = 15
    population_size=200, # Default = 33
    maxsize=30, # Limita a complexidade máxima das equações (Default = 20)
    ncycles_per_iteration=1100,
    binary_operators=["+", "*", "-", "/", "^"], # Defino os operadores binários que vou usar (já tem os mais importantes)
    unary_operators=[ # Defino os operadores unários que vou usar (já tem os mais importantes)
        "sin",
        "cos",
        "exp",
        "log",
        "sinh",
        "cosh",
        "erf",
    ],
    turbo=True, # Tende a acelerar o processo de treinamento, mas pode gerar erros em alguns casos
    bumper=True, # Mesmo funcionamento que o turbo
    warm_start=True, # Deixar sempre true para ter a possibilidade de continuar de onde parou
    nested_constraints={"sin": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}, "cos": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1},
                        "sinh": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}, "cosh": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1},
                        "erf": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}, "log": {"sin": 1, "cos": 1, "sinh": 1, "cosh": 1, "erf": 1, "log": 1}},
    constraints={"^": (9, 1)},
    annealing=True,
    progress=False, # Desativo a barra de progresso (Eu acho feio ¯\_(ツ)_/¯)
    verbosity=1, # Defina como 1 para mostrar todas as equações na hora do treinamento
    run_id="resultados", # Caminho para onde vai o modelo treinado
    elementwise_loss='LogitDistLoss()',
)
modelo.tournament_selection_n = int(modelo.population_size*0.303) # Default = 15 /tamanho do torneio
modelo.topn = int(modelo.population_size*0.303) # Default = 12 / elitismo


# surface (1 = Hard, 0 = clay)
# diff_hand (0 = mesma dominancia, 1 = vencedor destro e perdedor canhoto, -1 = vencedor canhoto e perdedor destro) 
# De resto, se positivo o vencedor tem mais que o perdedor. Se negativo, o contrário
# ht -> cm / age -> y
nome_X = ["diff_ht","diff_age","diff_rank","diff_hand","surface"]

modelo.fit(var_independentes, var_dependentes, variable_names=nome_X) # Treino o modelo

print(modelo)
