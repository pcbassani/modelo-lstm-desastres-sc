# Modelo LSTM para previsão de desastres em Santa Catarina

Modelo preditivo de desastres para os 293 municípios de Santa Catarina, treinado com dados diários de 2000 a 2024 (chuva, vento e radiação do ERA5, ONI da NOAA/CPC e registros de desastres). Uma rede LSTM estima, para cada município e dia, a chance de um desastre ser registrado nos próximos 3 dias. As previsões municipais também são agregadas em um alerta estadual.

## Resultados (período de teste, 2021–2024)

| Nível | Desastres detectados | Dias sem desastre reconhecidos | Acurácia |
| --- | --- | --- | --- |
| Municipal | 61,1% | 78,7% | 78,2% |
| Estadual (10+ municípios atingidos) | 57,8% | 57,7% | 57,8% |

Cada previsão também recebe um nível de risco e um comunicado:

| Nível de risco | Comunicado | Chance real de desastre (municipal) |
| --- | --- | --- |
| Baixo | Observação | 1,3% |
| Médio | Aviso | 2,4% |
| Alto e muito alto | Alerta | 9,6% (2,9× a média) |

## Estrutura

```
src/modelo/
├── modelo_lstm_desastres.py   # modelo LSTM: treino, avaliação, níveis de risco e comunicados
├── analise_limiares.py        # limiares de chuva a partir dos quais o risco de desastre dobra
├── importancia_variaveis.py   # testes estatísticos: qual variável pesa mais (logística multivariada)
└── climatologia.py            # calendário de risco por município e mês
```

## Como rodar

1. Instale as dependências:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Coloque o arquivo de dados `serie_precipitacao_com_desastres_enso_era5.csv` na raiz do repositório. Ele não está no repositório porque tem 363 MB.

3. Rode os scripts a partir da raiz do repositório:

   ```bash
   python3 src/modelo/modelo_lstm_desastres.py
   ```

   O modelo pede o arquivo do ONI no terminal: arraste o arquivo para a janela ou aperte Enter para baixá-lo da NOAA.

Os parâmetros ajustáveis ficam no topo de cada script, com comentários explicando o que cada um faz. As saídas são criadas em `resultados_sc/` (tabelas, gráficos e modelo treinado) e `analises_anteriores/` (a saída completa do terminal de cada execução).

## Métodos

- LSTM com ramo separado para as características fixas da amostra (inspirada na EA-LSTM, Kratzert et al., 2019).
- Variáveis hidrológicas: índice de precipitação antecedente (Kohler & Linsley, 1951) e SPI simplificado (McKee et al., 1993).
- Focal loss (Lin et al., 2017), peso de classes e calibração de Platt.
- Limiar de alerta adaptativo, escolhido na validação pelo critério de equilíbrio entre sensibilidade e especificidade.
- Divisão temporal: treino 2000–2016, validação 2017–2020 e teste 2021–2024.

## Limitações

O AUC do modelo (cerca de 0,75) é equivalente ao de uma climatologia simples por município e mês (0,76). Cerca de 59% dos desastres registrados ocorreram com chuva igual à de dias normais, o que limita o que variáveis meteorológicas conseguem prever. Separar os desastres hidrológicos das estiagens e incluir uma previsão do tempo como entrada são os próximos passos indicados.
