import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from implementação.shared import *

def plota_grafico():      
    # Inicializo o vetor
    valores_predicao = []

    dados_planilha = np.loadtxt(ARQUIVO_TESTE, delimiter = ',' , dtype = float, skiprows=1)  # Colunas F, G, H, I, K  

    # Faço a divisão entre as variáveis dependentes e independentes 
    var_independentes = dados_planilha[:, [0, 1, 2, 3, 4]]  # Contém somente X                        
    var_dependentes = dados_planilha[:, 5]  # Contém somente Y

    # Carrego o modelo já existente para a variável "modelo"
    modelo = PySRRegressor.from_file(run_directory=DIR_RESULTADOS)

    # Faço a contagem das equações
    quantidade_equacoes = int(modelo.equations_.count()["equation"])

    while True:
        valores_predicao = []
        while True:
            os.system('clear')
            print("+---------------------------------------------------------+")
            print("|                  O Regressor simbólico                  |")
            print("+---------------------------------------------------------+")
            print("|-> Escolha a equação de predição gerada pelo PySR        |")
            print("|-> As equações estão ordenadas por complexidade          |")
            print("|-> Deixe em branco e pressione Enter para usar a equação |")
            print("|  escolhida pelo PySR com base na sua seleção de modelo  |")
            print("+---------------------------------------------------------+\n")

            print(f"Encontrei {quantidade_equacoes} equações")
            string_numero_equacao = input(f"-> Selecione a equação e pressione Enter (0 para voltar): ")

            if string_numero_equacao == "":
                valores_predicao = modelo.predict(var_independentes)
                equacao = PySRRegressor.sympy(modelo)
                break

            numero_equacao = int(string_numero_equacao)
            numero_equacao -= 1

            if numero_equacao == -1:
                return

            elif 0 <= numero_equacao < quantidade_equacoes:
                valores_predicao = modelo.predict(var_independentes, numero_equacao)
                equacao = PySRRegressor.sympy(modelo, numero_equacao)
                break
            else:
                print("Essa equação não existe!")
                pausa_tela()

        # Calculando as predições binárias
        predicoes_binarias = (valores_predicao >= 0.5).astype(int)  # Usando o limiar de 0.5 para binarizar as predições

        from sklearn.metrics import accuracy_score
        acuracia = accuracy_score(var_dependentes, predicoes_binarias)
        print(f"Acurácia: {acuracia:.3f}")
        pausa_tela()

        # Plotando a matriz de confusão
        cm = confusion_matrix(var_dependentes, predicoes_binarias)

        # Estilo e formatação da matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.3)
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
                         xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'],
                         cbar_kws={'label': 'Contagem de Ocorrências'})

        plt.title('Matriz de Confusão', fontsize=18)
        plt.xlabel('Classe Predita', fontsize=14)
        plt.ylabel('Classe Real', fontsize=14)

        # Removendo as porcentagens e mantendo apenas os valores absolutos
        plt.tight_layout()
        plt.show()
        plt.savefig('matriz_confusao.png')  # Salvando a matriz de confusão

        # Plotando a Curva ROC
        fpr, tpr, thresholds = roc_curve(var_dependentes, valores_predicao)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Curva ROC - Performance do Modelo', fontsize=18)
        plt.xlabel('Taxa de Falsos Positivos', fontsize=14)
        plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=14)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        plt.savefig('curva_roc.png')  # Salvando a curva ROC

        break  # Sai do loop principal após gerar o gráfico
