from implementação.shared import *
from implementação.regressao_simbolica import *
from implementação.plotar import *
from implementação.metricas import *
from implementação.continuar_regressao import *
from pathlib import Path
while True:
    # Pequena interface =================================================
    os.system('clear')
    print("+---------------------------------------------------------+")
    print("|                  O Regressor simbólico                  |")
    print("+---------------------------------------------------------+")
    print("| 1. Usar a Regressão simbólica                           |")
    print("| 2. Continuar a regressão a partir de um checkpoint      |")
    print("| 3. Plotar o gráfico para comparar com a predição        |")
    print("| 4. Observar métricas de erro -> MSE, RMSE, MAE ...      |")
    print("| 5. Mostrar resultados                                   |")
    print("| 0. Sair                                                 |")
    print("+---------------------------------------------------------+")
    print("|-> Lembrar de alterar a constante com o arq de entrada!  |")
    print("|-> Lembrar de alterar as colunas de dados de entrada!    |")
    print("+---------------------------------------------------------+\n")

    caso = int(input("-> Selecione a opção: "))

    if caso == 0:
        # Encerra o programa
        print("Programa encerrado!\n")
        exit(0)

    else:                                                                             
        # Switch case para o usuário escolher o que quer fazer =================================================
        if caso == 1:                                                                     
            regressao_simbolica()
            
        elif caso == 2: 
            continua_regressao()

        elif caso == 3:                                                                  
            plota_grafico()

        elif caso == 4:                                                                      
            observa_metricas()

        elif caso == 5:
            print(f"\n{PySRRegressor.from_file(run_directory= DIR_RESULTADOS)}")
            pausa_tela()
            
        else:
            print("Opção inválida")
            pausa_tela()
