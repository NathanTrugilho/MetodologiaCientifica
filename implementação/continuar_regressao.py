from implementação.shared import *

def continua_regressao():
    
    limpa_tela()
    
    # Carrego os dados da planilha para a memória ==================
    dados_planilha = np.loadtxt(ARQUIVO_TREINO, delimiter = ',' , dtype = float, skiprows=1, usecols=[5,6,7,8,10]) # Colunas C, F, G, H, I, K  

    # Faço a divisão entre as variáveis dependentes e independentes 
    var_independentes = dados_planilha[ : ,  [0, 1, 2, 3]]  # Contém somente X                        
    var_dependentes = dados_planilha[ : , 4] # Contém somente Y

    # Carrego o estado do modelo para a memória ======
    try: 
        modelo = PySRRegressor.from_file(run_directory=DIR_RESULTADOS)
    except:
        print("\nNão existe um checkpoint a ser carregado!")
        pausa_tela()
        return

    if not os.path.isfile(ARQUIVO_CONTROLE):
        print("O arquivo de controle não existe!")
        pausa_tela()
        return

    # Defino o nome das variáveis independentes para mostrá-los durante a exibição das equações
    nome_var_independentes = ["Speed","Draft_Bow","Draft_Stern","Beaufort"]

    # Definir o tempo que o modelo vai executar 
    tempo_execucao = 5*DIA

    for checkpoint in range(tempo_execucao//HORA):
        print(f"Checkpoint {checkpoint} de {tempo_execucao//HORA}")
        modelo.fit(var_independentes,var_dependentes, variable_names= nome_var_independentes) # Treino o modelo

        # Atualizo o arquivo de controle
        with open(ARQUIVO_CONTROLE, "r+") as f:
            horas = int(f.read().strip()) + 1
            f.seek(0)
            f.write(str(horas))
            f.truncate()

    print(modelo)
    print("Treinamento finalizado!")
    pausa_tela()