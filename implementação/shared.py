from pysr import PySRRegressor
import os
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, mean_absolute_percentage_error
import pandas as pd

ARQUIVO_TREINO = "dados/treino.csv"
ARQUIVO_VALIDACAO = "dados/validacao.csv"
ARQUIVO_TESTE = "dados/teste.csv"

EQUACOES_PKL = "checkpoint.pkl"
DIR_RESULTADOS = "outputs/resultados"

# Arquivo que vai controlar o tempo de treino
ARQUIVO_CONTROLE = "horas_treino.txt"

HORA = 3600
DIA = 24*HORA
SEMANA = 7*DIA

INF = 10000000000000

def pausa_tela():
    input("Pressione 'Enter' para continuar...")

def limpa_tela():
    os.system("clear")
