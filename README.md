# Modelo LSTM para previsão de desastres em Santa Catarina

Modelo preditivo de desastres para os 293 municípios de Santa Catarina. Uma rede neural LSTM lê os últimos 30 dias de dados meteorológicos de cada município e estima a chance de um desastre ser registrado nos **próximos 3 dias**. As previsões municipais também são combinadas num alerta estadual (dias com 10 ou mais municípios atingidos).

## O modelo

**Dados:** série diária de 2000 a 2024 (2,6 milhões de linhas, uma por município por dia) com chuva, vento e radiação do ERA5, registros de desastres e o índice ONI do El Niño (NOAA/CPC, baixado pelo próprio script).

**Variáveis de entrada**, organizadas pelo papel físico de cada uma:

| Grupo | Variáveis |
| --- | --- |
| Gatilho (chuva intensa recente) | chuva do dia, chuva acumulada em 3 dias, dias de chuva forte |
| Umidade do solo | índice de precipitação antecedente (API), chuva acumulada em 30 dias, dias de chuva em 15 dias |
| Seca | dias secos consecutivos, SPI de 30 e 90 dias |
| Atmosfera | radiação solar, vento máximo |
| Contexto | localização, época do ano, fase do El Niño, risco histórico do município no mês |

**Arquitetura:** a LSTM lê a sequência diária, e o contexto entra por um ramo separado que define o estado inicial da rede (inspirado na EA-LSTM, Kratzert et al., 2019). O treino usa focal loss e peso de classes, porque os desastres ocorrem em só 3% dos dias. A divisão é temporal: treino de 2000 a 2016, validação de 2017 a 2020 e teste de 2021 a 2024.

**Saída:** para cada município e dia, uma probabilidade, um **nível de risco** e um **comunicado**:

| Nível de risco | Comunicado | Chance real de desastre (teste) |
| --- | --- | --- |
| Baixo | Observação | 1,3% |
| Médio | Aviso | 2,4% |
| Alto e muito alto | Alerta | 9,6% (2,9× a média) |

**Resultados no período de teste (2021–2024):**

| Nível | Desastres detectados | Dias sem desastre reconhecidos | Acurácia |
| --- | --- | --- | --- |
| Municipal | 61,1% | 78,7% | 78,2% |
| Estadual | 57,8% | 57,7% | 57,8% |

O relatório completo, com a explicação do modelo, dos alertas, dos parâmetros ajustados, das análises estatísticas e de todos os gráficos, está em [`docs/relatorios/relatorio_modelo_lstm.md`](docs/relatorios/relatorio_modelo_lstm.md).

## Estrutura do repositório

```
├── README.md
├── requirements.txt
├── src/
│   ├── modelo/
│   │   ├── modelo_lstm_desastres.py   # o modelo: treino, previsão, níveis de risco e comunicados
│   │   └── graficos_previsoes.py      # gráficos das previsões (gerados a cada execução)
│   └── analises/
│       ├── analises_limiares.py       # limiares de chuva a partir dos quais o risco dobra
│       ├── importancia_variaveis.py   # testes estatísticos de importância das variáveis
│       └── climatologia.py            # calendário de risco por município e mês
└── docs/
    ├── analises/
    │   └── graficos/                  # gráficos em PNG das previsões e das análises
    └── relatorios/
        └── relatorio_modelo_lstm.md   # relatório do trabalho
```

## Como rodar na máquina local

**1. Requisitos:** Python 3.9 ou mais recente, com TensorFlow. No Mac usado no projeto, o Python do sistema (`/usr/bin/python3`) funcionou; o TensorFlow do Anaconda travava ao ser importado.

**2. Baixe o código:**

```bash
git clone https://github.com/pcbassani/modelo-lstm-desastres-sc.git
```

```bash
cd modelo-lstm-desastres-sc
```

**3. Instale as dependências:**

```bash
python3 -m pip install -r requirements.txt
```

**4. Coloque os dados na raiz do repositório.** O arquivo `serie_precipitacao_com_desastres_enso_era5.csv` não está no GitHub porque tem 363 MB. O CSV usa `;` como separador e `,` como decimal.

**5. Rode o modelo**, sempre a partir da raiz do repositório:

```bash
python3 src/modelo/modelo_lstm_desastres.py
```

Durante a execução, o script pede o arquivo do ONI: arraste o arquivo para a janela do terminal e aperte Enter, ou só aperte Enter para baixá-lo da NOAA. Na primeira execução, o CSV é lido e salvo em cache, o que demora alguns minutos; o treino completo pode levar mais de uma hora.

**6. Onde ficam os resultados:**

| Pasta | Conteúdo |
| --- | --- |
| `docs/analises/graficos/` | Gráficos em PNG das previsões (matrizes de confusão, níveis de risco, curvas ROC, alertas no tempo) |
| `resultados_sc/` | Tabelas das previsões, métricas por município, modelo treinado e cache dos dados |
| `analises_anteriores/` | A saída completa do terminal de cada execução, com os parâmetros usados |

**Outros comandos:**

```bash
python3 src/modelo/graficos_previsoes.py
```

Refaz os gráficos a partir das previsões já salvas, sem treinar de novo.

```bash
python3 src/analises/analises_limiares.py
```

```bash
python3 src/analises/importancia_variaveis.py
```

```bash
python3 src/analises/climatologia.py
```

As análises estatísticas reaproveitam os dados e os parâmetros do modelo e levam poucos minutos.

**Parâmetros:** todos ficam no topo de `src/modelo/modelo_lstm_desastres.py`, com um comentário explicando o que cada um faz e como ajustar.

## Limitações

O AUC do modelo (0,75) é equivalente ao de uma climatologia simples por município e mês (0,76): boa parte do que o modelo aprende é **onde** e **em que época** os desastres costumam ser registrados. Cerca de 59% dos desastres ocorreram com chuva igual à de dias normais, o que limita o que variáveis meteorológicas conseguem prever. Separar os desastres hidrológicos das estiagens e incluir uma previsão do tempo como entrada são os próximos passos indicados.
