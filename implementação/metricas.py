from implementação.shared import *

def observa_metricas():

    dados_planilha = np.loadtxt(ARQUIVO_TESTE, delimiter = ',' , dtype = float, skiprows=1, usecols=[5,6,7,8,10]) # Colunas C, F, G, H, I, K  

    # Faço a divisão entre as variáveis dependentes e independentes 
    var_independentes = dados_planilha[ : ,  [0, 1, 2, 3]]  # Contém somente X                        
    var_dependentes = dados_planilha[ : , 4] # Contém somente Y

    modelo = PySRRegressor.from_file(run_directory=DIR_RESULTADOS)

    dicionario_mse = {}
    dicionario_rmse = {}
    dicionario_mae = {}
    dicionario_r2 = {}
    dicionario_median_abs_err = {}
    dicionario_mean_abs_perc_err = {}

    # Faço a contagem das equações
    quantidade_equacoes = int(modelo.equations_.count()["equation"])

    print("") #pular linha

    calcula_mse_geral(dicionario_mse, quantidade_equacoes, modelo, var_independentes, var_dependentes)

    calcula_rmse_geral(dicionario_rmse, quantidade_equacoes, modelo, var_independentes, var_dependentes)

    calcula_mae_geral(dicionario_mae, quantidade_equacoes, modelo, var_independentes, var_dependentes)

    calcula_r2_geral(dicionario_r2, quantidade_equacoes, modelo, var_independentes, var_dependentes)

    calcula_median_abs_err(dicionario_median_abs_err, quantidade_equacoes, modelo, var_independentes, var_dependentes)

    calcula_mean_abs_perc_err(dicionario_mean_abs_perc_err, quantidade_equacoes, modelo, var_independentes, var_dependentes)

    dados = {
        'Equações': list(PySRRegressor.sympy(modelo, i) for i in range (0, quantidade_equacoes)),
        'MSE': list(dicionario_mse.values()),
        'RMSE': list(dicionario_rmse.values()),
        'MAE': list(dicionario_mae.values()),
        'R2': list(dicionario_r2.values()),
        'Median absolute error': list(dicionario_median_abs_err.values()),
        'Mean absolute percentage error': list(dicionario_mean_abs_perc_err.values())
    }

    df = pd.DataFrame(dados)

    # Crio o arquivo do tipo excel
    df.to_excel('metricas_equacoes.xlsx', index=False)
    pausa_tela()

def calcula_mse_geral(dicionario_mse, quantidade_equacoes, modelo, xVector, yVector):
    
    # Calculo o MSE de todas as equações e guardo num dicionário
    for i in range(0, quantidade_equacoes):

        valores_predicao = modelo.predict(xVector, i)
        mse = mean_squared_error(yVector, valores_predicao)
        dicionario_mse[f"Eq{i}:"] = mse

    # Digo que a primeira chave tem o menor valor de MSE
    chave_menor_valor = next(iter(dicionario_mse.keys()))
    menor_valor = dicionario_mse[chave_menor_valor]
    
    for chave, valor in dicionario_mse.items():
        #print(chave, valor)
        if valor < menor_valor:
            chave_menor_valor = chave
            menor_valor = valor
    
    # Mostro a melhor equação (com menor MSE)
    print(f"A {chave_menor_valor} tem o menor MSE com o valor de: {dicionario_mse[chave_menor_valor]}\n")

def calcula_rmse_geral(dicionario_rmse, quantidade_equacoes, modelo, xVector, yVector):
    
    # Calculo o RMSE de todas as equações e guardo num dicionário
    for i in range(0, quantidade_equacoes):

        valores_predicao = modelo.predict(xVector, i)
        rmse = np.sqrt(mean_squared_error(yVector, valores_predicao)) # Tiro a raiz quadrada do MSE
        dicionario_rmse[f"Eq{i + 1}:"] = rmse

    # Digo que a primeira chave tem o menor valor de RMSE
    chave_menor_valor = next(iter(dicionario_rmse.keys()))
    menor_valor = dicionario_rmse[chave_menor_valor]
    
    for chave, valor in dicionario_rmse.items():
        #print(chave, valor)
        if valor < menor_valor:
            chave_menor_valor = chave
            menor_valor = valor
    
    # Mostro a melhor equação (com menor RMSE)
    print(f"A {chave_menor_valor} tem o menor RMSE com o valor de: {dicionario_rmse[chave_menor_valor]}\n")

def calcula_mae_geral(dicionario_mae, quantidade_equacoes, modelo, xVector, yVector):
    
    # Calculo o MAE de todas as equações e guardo num dicionário
    for i in range(0, quantidade_equacoes):

        valores_predicao = modelo.predict(xVector, i)
        mae = mean_absolute_error(yVector, valores_predicao)
        dicionario_mae[f"Eq{i + 1}:"] = mae

    # Digo que a primeira chave tem o menor valor de MAE
    chave_menor_valor = next(iter(dicionario_mae.keys()))
    menor_valor = dicionario_mae[chave_menor_valor]
    
    for chave, valor in dicionario_mae.items():
        #print(chave, valor)
        if valor < menor_valor:
            chave_menor_valor = chave
            menor_valor = valor
    
    # Mostro a melhor equação (com menor MAE)
    print(f"A {chave_menor_valor} tem o menor MAE com o valor de: {dicionario_mae[chave_menor_valor]}\n")

def calcula_r2_geral(dicionario_r2, quantidade_equacoes, modelo, xVector, yVector):

    # Calculo o R2 de todas as equações e guardo num dicionário
    for i in range(0, quantidade_equacoes):

        valores_predicao = modelo.predict(xVector, i)
        r2 = r2_score(yVector, valores_predicao)
        dicionario_r2[f"Eq{i + 1}:"] = r2

    # Digo que a primeira chave tem o menor valor de R2
    chave_maior_valor = next(iter(dicionario_r2.keys()))
    maior_valor = dicionario_r2[chave_maior_valor]

    for chave, valor in dicionario_r2.items():
        
        if valor > maior_valor:
            chave_maior_valor = chave
            maior_valor = valor

    # Mostro a melhor equação (com menor R2)
    print(f"A {chave_maior_valor} tem o maior R2 com o valor de: {dicionario_r2[chave_maior_valor]}\n")

def calcula_median_abs_err(dicionario_median_abs_err, quantidade_equacoes, modelo, xVector, yVector):

    for i in range(0, quantidade_equacoes):

        valores_predicao = modelo.predict(xVector, i)
        median_abs_err = median_absolute_error(yVector, valores_predicao)
        dicionario_median_abs_err[f"Eq{i + 1}:"] = median_abs_err

    chave_menor_valor = next(iter(dicionario_median_abs_err.keys()))
    menor_valor = dicionario_median_abs_err[chave_menor_valor]

    for chave, valor in dicionario_median_abs_err.items():
        if valor < menor_valor:
            chave_maior_valor = chave
            menor_valor = valor

    print(f"A {chave_maior_valor} tem o menor Median absolute error com o valor de: {dicionario_median_abs_err[chave_maior_valor]}\n")

def calcula_mean_abs_perc_err(dicionario_mean_abs_perc_err, quantidade_equacoes, modelo, xVector, yVector):

    for i in range(0, quantidade_equacoes):

        valores_predicao = modelo.predict(xVector, i)
        mean_abs_perc_err = mean_absolute_percentage_error(yVector, valores_predicao)
        dicionario_mean_abs_perc_err[f"Eq{i + 1}:"] = mean_abs_perc_err

    chave_menor_valor = next(iter(dicionario_mean_abs_perc_err.keys()))
    menor_valor = dicionario_mean_abs_perc_err[chave_menor_valor]

    for chave, valor in dicionario_mean_abs_perc_err.items():
        if valor < menor_valor:
            chave_maior_valor = chave
            menor_valor = valor

    print(f"A {chave_maior_valor} tem o menor Mean absolute percentage error com o valor de: {dicionario_mean_abs_perc_err[chave_maior_valor]}\n")