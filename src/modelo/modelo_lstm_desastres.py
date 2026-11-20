"""
Resultados gerados:
  1. Por município: probabilidade diária de desastre em cada município.
  2. Para o estado: agregação das probabilidades municipais, que dá a chance
     de pelo menos MINIMO_MUNICIPIOS_EVENTO_ESTADUAL municípios serem atingidos
     no dia e o número esperado de municípios atingidos.

Cada município também recebe um identificador aprendido (embedding), que
permite ao modelo captar características locais (relevo, rios, ocupação) que
as variáveis meteorológicas não mostram.

Dependências:
    python3 -m pip install numpy pandas scikit-learn tensorflow matplotlib openpyxl
"""

import hashlib
import os
import sys
import traceback
import urllib.request
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from graficos_previsoes import gerar_graficos_previsoes

CAMINHO_DADOS = "serie_precipitacao_com_desastres_enso_era5.csv"

SEPARADOR_CSV = ";"

DECIMAL_CSV = ","

FORMATO_DATA = "%Y-%m-%d"

PERCENTUAL_DADOS = 100

MUNICIPIOS_FILTRO = None

LINHAS_POR_BLOCO = 500_000

USAR_CACHE = True

MINIMO_HORAS_ERA5 = 20

LIMIAR_DESASTRES = 1

HORIZONTE_PREVISAO = 3

ALVO_JANELA_HORIZONTE = True

# O que o modelo aprende a prever:
#   "desastre"      = registro de desastre nos próximos HORIZONTE_PREVISAO dias.
#   "chuva_extrema" = chuva extrema nos próximos HORIZONTE_PREVISAO dias (soma da
#                     chuva desses dias acima do limiar abaixo). O registro de
#                     desastre vira uma TAG de avaliação: no fim, o script mede
#                     quantas vezes mais desastres acontecem nos dias com alerta
#                     de chuva extrema. A chuva é mais previsível pelas variáveis
#                     meteorológicas do que o registro de desastre.
ALVO = "desastre"

# Como definir "chuva extrema" (só com ALVO = "chuva_extrema"):
#   "percentil_municipio" = acima do percentil PERCENTIL_CHUVA_EXTREMA da chuva
#                           de HORIZONTE_PREVISAO dias do próprio município
#                           (calculado no treino). Respeita o clima de cada
#                           região: o que é extremo no Oeste é diferente da Serra.
#   "absoluto"            = acima de LIMIAR_CHUVA_MM em todos os municípios.
TIPO_LIMIAR_CHUVA = "percentil_municipio"

# Percentil usado em "percentil_municipio". 0.95 = os 5% de períodos mais
# chuvosos de cada município. Maior = eventos mais raros e mais extremos.
# Sugestão de teste: 0.90, 0.95, 0.98.
PERCENTIL_CHUVA_EXTREMA = 0.95

# Limiar em mm usado em "absoluto". 46 mm em 3 dias foi o valor a partir do qual
# o risco de desastre dobra na análise de limiares (analises_limiares.py).
LIMIAR_CHUVA_MM = 46.0

# Se True, o modelo prevê só o INÍCIO de cada evento. Muitos desastres ficam
# registrados por vários dias seguidos (ex.: um decreto que dura semanas), e os
# dias depois do primeiro não dependem da chuva daquele dia. Contar só o
# primeiro dia deixa a tarefa coerente com o que as variáveis conseguem explicar.
# False = volta ao comportamento anterior (todo dia com registro é evento).
APENAS_INICIO_EVENTO = False

# Quantos dias sem registro são necessários para um novo registro contar como
# um NOVO evento. Ex.: 3 = registros com até 3 dias de intervalo são tratados
# como o mesmo evento. Valores maiores juntam mais registros num evento só.
# Sugestão de teste: 1, 3, 7.
DIAS_ENTRE_EVENTOS = 3

# Se True, os dias de continuação de um evento (registrados, mas que não são o
# início) ficam FORA do treino e da avaliação, em vez de contarem como "sem
# evento". Isso evita ensinar ao modelo que um dia de chuva forte durante um
# desastre em andamento é um dia normal. Só tem efeito com APENAS_INICIO_EVENTO.
IGNORAR_CONTINUACAO_EVENTO = True

# Variáveis numéricas da planilha usadas como entrada. Colunas da lista que não
# existirem na planilha são ignoradas (o terminal avisa quais ficaram de fora).
# Retirar uma variável por vez e comparar os resultados mostra quais delas
# realmente ajudam a prever. As colunas do ONI (abaixo) entram automaticamente.
COLUNAS_ENTRADA = [
    "precipitacao",
    "chuva_acum_3d",
    "chuva_acum_5d",
    "chuva_acum_7d",
    "dias_chuva_3d",
    "dias_chuva_7d",
    "dias_chuva_15d",
    "dias_chuva_p90_7d",
    "dias_chuva_p90_15d",
    "vento_medio_ms",
    "vento_maximo_ms",
    "radiacao_kwh_m2",
]

# -----------------------------------------------------------------------------
# ONI (Oceanic Niño Index) oficial da NOAA / CPC
# -----------------------------------------------------------------------------
# O ONI mede o El Niño / La Niña: é a anomalia da temperatura do Pacífico
# (região Niño 3.4), em média móvel de 3 meses. El Niño costuma trazer mais
# chuva ao Sul do Brasil. Ele não está na planilha: o script lê o arquivo do
# ONI e junta a cada dia o valor do mês correspondente.

# Se True, o ONI entra como variável de entrada do modelo.
USAR_ONI = True

# Caminho do arquivo do ONI já baixado. Com None, o script PERGUNTA no terminal
# onde está o arquivo (basta arrastar o arquivo para a janela do terminal e
# apertar Enter). Se você apertar Enter sem informar nada, ele tenta baixar o
# arquivo direto da NOAA. Formatos aceitos:
#   - oni.ascii.txt da CPC (colunas SEAS YR TOTAL ANOM)
#     https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
#   - oni.data da NOAA PSL (ano + 12 meses por linha)
#     https://psl.noaa.gov/data/correlation/oni.data
# Ex.: CAMINHO_ONI = "/Users/vinicius/Downloads/oni.ascii.txt"
CAMINHO_ONI = None

# Endereço usado para baixar o ONI quando nenhum arquivo é informado.
URL_ONI_NOAA = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Defasagens do ONI em meses. Para cada número vira uma coluna: 1 = ONI do mês
# anterior ("ONI_lag1"), 3 = de 3 meses antes etc. O efeito do El Niño sobre a
# chuva costuma aparecer com alguns meses de atraso.
# Atenção: 0 (o ONI do próprio mês) usa a média de 3 meses CENTRADA no mês, ou
# seja, inclui o mês seguinte, que ainda não seria conhecido no momento da
# previsão. Para uma avaliação honesta, use só defasagens de 1 ou mais.
# Sugestão de teste: [1], [1, 3], [1, 3, 6], [1, 2, 3, 6, 12].
DEFASAGENS_ONI = [1]

# Valor do ONI (em °C) que separa as fases: El Niño (ONI >= limiar),
# La Niña (ONI <= -limiar) ou Neutro. A NOAA usa 0,5. É uma versão simplificada:
# a definição oficial exige 5 trimestres seguidos acima do limiar.
LIMIAR_FASE_ENSO = 0.5

# Se True, antes do treino o script compara quantos desastres acontecem em
# cada fase (El Niño, La Niña, Neutro): taxa de dias com evento, risco relativo
# em relação à fase Neutra e teste qui-quadrado de independência. Aqui a fase
# é a do próprio mês (análise descritiva, não previsão). Resultado no terminal,
# em PASTA_SAIDA/desastres_por_fase_enso.csv e desastres_por_fase_enso.png.
ANALISAR_DESASTRES_POR_FASE = True

# A "tag" do ENSO no modelo usa a fase do ONI da MENOR defasagem de
# DEFASAGENS_ONI (com [1], a fase do mês anterior, já conhecida no dia).
# Se True, cada fase vira uma coluna 0/1 de entrada (fase_El Niño etc.).
CRIAR_FASE_ENSO = True

# Se True, associa a tag ao histórico de desastres: para cada município e fase,
# calcula no período de TREINO quanto o risco de desastre naquela fase difere
# do risco médio do município (coluna "risco_fase_municipio"; 1 = igual à
# média, 1,5 = 50% mais desastres naquela fase). Assim o modelo recebe
# diretamente "este município costuma ter mais desastres em El Niño".
USAR_RISCO_HISTORICO_FASE = True

# Suavização do risco histórico, em dias: municípios com poucos dias numa fase
# ficam mais perto de 1 (sem efeito), para não confundir acaso com padrão.
# Valores maiores = mais conservador. Sugestão de teste: 90, 365, 1000.
SUAVIZACAO_RISCO_FASE = 365

# Coluna de texto com a fase do ENSO usada pelo modelo.
COLUNA_FASE_ENSO = "fase_ENSO"

# -----------------------------------------------------------------------------
# Climatologia: risco histórico de cada município em cada mês do ano
# -----------------------------------------------------------------------------
# Se True, o modelo recebe duas variáveis calculadas no período de TREINO a
# partir do próprio alvo (desastre ou chuva extrema, conforme ALVO):
#   "clima_prob"  = probabilidade histórica do evento no município naquele mês;
#   "clima_risco" = essa probabilidade dividida pela média anual do município
#                   (1 = mês comum, 2 = o dobro da média, 0,5 = metade).
# É o "calendário de risco" do município dito de forma direta ao modelo.
USAR_CLIMATOLOGIA = True

# Suavização, em dias: meses com poucos dias de histórico ficam mais perto da
# média do município, para não confundir acaso com padrão.
# Sugestão de teste: 30, 100, 300.
SUAVIZACAO_CLIMATOLOGIA = 100

# Cortes do risco relativo para as categorias usadas no climatologia.py:
# baixa (abaixo do primeiro), média (entre os dois) e alta (acima do segundo).
CORTES_CATEGORIA_CLIMATOLOGIA = (0.75, 1.25)

COLUNAS_LOG = ["precipitacao", "chuva_acum_3d", "chuva_acum_5d", "chuva_acum_7d",
               "chuva_acum_15d", "chuva_acum_30d", "chuva_acum_90d", "api"]

# -----------------------------------------------------------------------------
# Variáveis hidrológicas (umidade do solo e seca), criadas a partir da chuva
# -----------------------------------------------------------------------------
# Índice de Precipitação Antecedente (API; Kohler & Linsley, 1951):
#   API(hoje) = K_API x API(ontem) + chuva(hoje)
# Mede a umidade acumulada no solo, dando mais peso à chuva recente. K perto de
# 1 = memória longa (solo demora a secar). Valores usuais: 0,85 a 0,95.
K_API = 0.90

# Chuva abaixo deste valor (mm) conta como "dia seco" na contagem de dias
# consecutivos sem chuva.
LIMIAR_DIA_SECO_MM = 1.0

# Janelas (em dias) do índice de seca/excesso de chuva. Para cada uma é criada
# a coluna "spi_<n>d": a chuva acumulada nos últimos n dias comparada com o
# normal do município naquele mês, em desvios padrão (versão simplificada do
# SPI de McKee et al., 1993). Negativo = mais seco que o normal (estiagem);
# positivo = mais chuvoso. Calculado só com o período de treino.
JANELAS_SPI = [30, 90]

# -----------------------------------------------------------------------------
# Organização das variáveis e arquitetura
# -----------------------------------------------------------------------------
# "separada" = a LSTM lê só a sequência diária (grupos abaixo) e as
#              características fixas da amostra (município, localização, época,
#              ENSO, climatologia) entram por um ramo próprio que define o estado
#              inicial da LSTM e se junta à saída dela. Inspirado na EA-LSTM
#              (Kratzert et al., 2019) e no Temporal Fusion Transformer
#              (Lim et al., 2021), que separam entradas dinâmicas e estáticas.
# "simples"  = arquitetura anterior: todas as variáveis entram na sequência.
ARQUITETURA = "separada"

# Grupos de variáveis da sequência diária (usados com ARQUITETURA = "separada").
# Cada grupo tem um papel físico; variáveis que não existirem são ignoradas.
GRUPO_GATILHO = ["precipitacao", "chuva_acum_3d", "dias_chuva_p90_7d"]       # chuva intensa recente
GRUPO_UMIDADE = ["api", "chuva_acum_30d", "dias_chuva_15d"]                  # solo saturado
GRUPO_SECA = ["dias_secos_consecutivos", "spi_30d", "spi_90d"]               # estiagem
GRUPO_ATMOSFERA = ["radiacao_kwh_m2", "vento_maximo_ms"]                     # céu encoberto, vento

# Neurônios da camada que codifica as características fixas da amostra.
UNIDADES_CONTEXTO = 16

# Chuva acumulada de prazo mais longo, calculada a partir de "precipitacao".
# Cada número vira uma coluna (ex.: 30 = "chuva_acum_30d", soma dos últimos 30
# dias). Semanas de chuva saturam o solo e antecedem deslizamentos e cheias.
# Use [] para não criar. Sugestão de teste: [15, 30], [30, 60].
CHUVA_ACUMULADA_LONGA = [15, 30]

# Colunas de chuva que ganham uma versão "em relação ao normal do município":
# o percentil do valor no histórico do próprio município (0 = o menor já
# visto, 0,99 = mais que 99% dos dias). 80 mm é comum na Serra e extremo no
# Oeste; o percentil deixa municípios diferentes comparáveis. Calculado só com
# o período de treino. Use [] para não criar.
COLUNAS_PERCENTIL_MUNICIPIO = ["precipitacao", "chuva_acum_3d", "chuva_acum_7d", "chuva_acum_30d"]

USAR_COORDENADAS = True

USAR_SAZONALIDADE = True

USAR_EMBEDDING_MUNICIPIO = True

DIMENSAO_EMBEDDING = 4

# Tipo de normalização: "standarxd" (média 0, desvio 1) ou "minmax" (escala 0-1).
TIPO_NORMALIZACAO = "standard"

DATA_FIM_TREINO = "2016-12-31"        # ex.: "2016-12-31"
DATA_FIM_VALIDACAO = "2020-12-31"     # ex.: "2019-12-31"

PROPORCAO_TREINO = 0.7
PROPORCAO_VALIDACAO = 0.15

# Sugestão de teste: 7, 15, 30. Com 30 dias a sequência cobre o ciclo de
# umidade do solo (a chuva de 30 dias foi a variável mais importante).
TAMANHO_JANELA = 30


# Sugestão de teste: [32], [64], [64, 32].
UNIDADES_LSTM = [64]

# Lê a janela nos dois sentidos. Às vezes melhora, mas dobra o custo.
BIDIRECIONAL = False

# Fração de neurônios desligados aleatoriamente após cada camada (0.0 a 0.5).
# Aumente se a validação piorar enquanto o treino melhora (overfitting).
DROPOUT = 0.5

# Dropout dentro da célula LSTM. 0.0 treina mais rápido; teste 0.1-0.2
# se houver overfitting.
DROPOUT_RECORRENTE = 0.0

# Neurônios da camada densa antes da saída. Use 0 para remover a camada.
UNIDADES_DENSA = 32

# Penalidade para pesos grandes (regularização L2). Ex.: 0.0, 1e-4, 1e-3.
# Aumente se houver overfitting.
REGULARIZACAO_L2 = 1e-3

# Sugestão de teste: 5e-4, 1e-3, 2e-3, 3e-3.
TAXA_APRENDIZADO = 2e-4

REDUZIR_TAXA_AUTOMATICAMENTE = True

TAMANHO_LOTE = 1024

# Número máximo de épocas. O early stopping costuma parar antes.
EPOCAS = 50

# Épocas sem melhora na validação antes de interromper o treino.
PACIENCIA_EARLY_STOPPING = 8

METRICA_MONITORADA = "val_pr_auc"

# probabilidades depois. Sugestão de teste: 0.1, 0.25, 0.5, 1.0.
FRACAO_NEGATIVOS_TREINO = 0.25

# desequilíbrio que ainda restar após a amostragem acima.
USAR_PESO_CLASSES = True

# Função de perda:
#   "binary_crossentropy" = padrão para classificação binária.
#   "focal"               = focal loss (Lin et al., 2017): diminui o peso dos
#                           exemplos fáceis (a maioria dos dias sem evento) e
#                           concentra o aprendizado nos casos difíceis.
FUNCAO_PERDA = "focal"

# Intensidade do foco nos casos difíceis da focal loss. 0 = igual à
# binary_crossentropy; 2 = valor do artigo original. Sugestão: 1, 2, 3.
GAMMA_FOCAL = 2.0

CALIBRAR_PROBABILIDADES = True

# Um número (ex.: 0.3) fixa o limiar: menor = detecta mais eventos e dá mais
# alarmes falsos; maior = menos alarmes falsos e mais eventos perdidos.
LIMIAR_MUNICIPIO = "auto"

LIMIAR_ESTADO = "auto"

# Critério usado para escolher o limiar quando LIMIAR_MUNICIPIO ou
# LIMIAR_ESTADO = "auto". O limiar é sempre escolhido nos dados de validação.
#   "f1"            = maximiza o F1 da classe evento. Bom quando eventos são
#                     raros, mas quando eles são maioria tende a alertar sempre.
#   "youden"        = maximiza (recall dos eventos + recall dos não eventos - 1).
#                     Obriga a acertar as duas classes; alertar sempre dá zero.
#   "recall_minimo" = garante pelo menos RECALL_MINIMO de detecção e, entre os
#                     limiares que garantem isso, escolhe o de maior precisão.
#   "equilibrio"    = maximiza a PIOR das duas taxas de acerto (sensibilidade e
#                     especificidade), deixando as duas o mais iguais possível.
#
# Para o objetivo de acertar pelo menos 50% dos desastres E 50% dos dias sem
# desastre, "youden" (Youden, 1950; Fluss et al., 2005) e "equilibrio" são os
# indicados. "equilibrio" alerta mais dias que "youden" e, por isso, mantém a
# sensibilidade acima de 50% mesmo quando o período de teste tem mais eventos
# que o de validação (o que acontece nos dados de SC).
CRITERIO_LIMIAR_MUNICIPIO = "equilibrio"
CRITERIO_LIMIAR_ESTADO = "equilibrio"

# Recall mínimo usado pelo critério "recall_minimo" (0 a 1). Ex.: 0.5 = o
# alerta precisa detectar pelo menos 50% dos eventos da validação. Valores
# maiores detectam mais desastres, mas geram mais alarmes falsos (a precisão
# e a acurácia caem). Sugestão de teste: 0.3, 0.5, 0.7.
RECALL_MINIMO = 0.5

# Se True, cada município recebe seu próprio limiar de alerta, escolhido na
# validação com CRITERIO_LIMIAR_MUNICIPIO. Isso ajuda municípios cujas
# probabilidades ficam sempre abaixo do limiar geral (e que por isso nunca
# recebiam alerta). Municípios com poucos eventos na validação usam o limiar
# geral, porque um limiar escolhido com poucos casos seria pouco confiável.
LIMIAR_POR_MUNICIPIO = True

# Mínimo de eventos na validação para um município ter limiar próprio.
# Sugestão de teste: 10, 20, 40.
MINIMO_EVENTOS_LIMIAR_MUNICIPIO = 20

# Quantos municípios precisam ser atingidos no mesmo dia para contar como um
# evento ESTADUAL. Com 1, quase todo dia tem desastre em algum lugar de SC e a
# métrica estadual perde o sentido. Valores maiores representam eventos de
# grande escala (ex.: chuvas que atingem várias regiões ao mesmo tempo).
# Considere a quantidade de municípios usada (PERCENTUAL_DADOS): com menos
# municípios, use valores menores. Sugestão de teste: 2, 3, 5, 10.
MINIMO_MUNICIPIOS_EVENTO_ESTADUAL = 10

# Se True, o alerta ESTADUAL usa um limiar adaptativo: a previsão do dia é
# comparada com as previsões dos últimos JANELA_LIMIAR_ESTADO_DIAS dias, e o
# alerta dispara quando ela fica acima de um percentil escolhido na validação.
# Assim o limiar acompanha mudanças ao longo do tempo (os registros de
# desastre aumentaram muito a partir de 2021) sem usar os resultados do teste:
# só previsões passadas entram na comparação. Não altera o nível municipal.
LIMIAR_ESTADO_ADAPTATIVO = True

# Quantos dias de previsões passadas formam a referência do limiar adaptativo.
# 365 = um ano inteiro, que inclui todas as estações.
JANELA_LIMIAR_ESTADO_DIAS = 365

# Mínimo de dias de histórico para um dia entrar na escolha do percentil (os
# primeiros dias da validação ainda não têm previsões passadas suficientes).
MINIMO_HISTORICO_LIMIAR_ESTADO = 60

# Se True, o alerta MUNICIPAL também usa limiar adaptativo: a probabilidade de
# cada município-dia é comparada com TODAS as previsões (de todos os
# municípios) dos últimos JANELA_LIMIAR_MUNICIPIO_DIAS dias, e o alerta
# dispara acima de um percentil escolhido na validação. Substitui os limiares
# por município, que se ajustavam mal quando o teste tinha mais eventos que a
# validação (os municípios menores chegavam a alertar em metade dos dias).
LIMIAR_MUNICIPIO_ADAPTATIVO = True
JANELA_LIMIAR_MUNICIPIO_DIAS = 365

# Níveis de risco, pela posição da previsão entre as previsões do último ano.
# Cortes = percentis que separam os níveis. (0.50, 0.80, 0.95) =
#   baixo (50% dos dias de menor risco), médio (próximos 30%),
#   alto (próximos 15%) e muito alto (5% dos dias de maior risco).
# É a forma usada em sistemas de alerta (níveis em vez de sim/não). Teste
# outros cortes, ex.: (0.60, 0.90, 0.99) para um "muito alto" mais raro.
CORTES_NIVEIS_RISCO = (0.50, 0.80, 0.95)
NOMES_NIVEIS_RISCO = ("baixo", "médio", "alto", "muito alto")

# Comunicado emitido em cada nível de risco, como nos sistemas de alerta:
#   observação = acompanhar a situação (risco baixo);
#   aviso      = atenção, condições podem piorar (risco médio);
#   alerta     = risco elevado, preparar resposta (riscos alto e muito alto).
COMUNICADO_POR_NIVEL = {"baixo": "observação", "médio": "aviso", "alto": "alerta", "muito alto": "alerta"}
ORDEM_COMUNICADOS = ("observação", "aviso", "alerta")

# Semente aleatória, para que os resultados possam ser reproduzidos.
SEMENTE = 42

# Pasta onde serão salvos o modelo, os gráficos e as tabelas de resultados.
PASTA_SAIDA = "resultados_sc"

# Pasta dos gráficos em PNG que vão para a documentação do repositório
# (previsões do teste, treinamento e análise por fase do ENSO).
PASTA_GRAFICOS = os.path.join("docs", "analises", "graficos")

# Pasta onde fica salva toda a saída do terminal de cada execução. Cada vez que
# o script roda, cria um arquivo novo com data e hora no nome
# (ex.: analises_2026-10-30_10-15-02.txt), com os parâmetros usados no topo.
# Assim dá para comparar execuções antigas sem perder nenhum resultado.
PASTA_ANALISES = "analises_anteriores"

COLUNA_MUNICIPIO = "municipio"
COLUNA_DATA = "data"
COLUNA_DESASTRES = "Desastres"
COLUNA_HORAS = "horas_com_dados_era5"
COLUNAS_COORDENADAS = ["latitude_municipio", "longitude_municipio"]

CONSTANTES_FIXAS = {"COLUNA_MUNICIPIO", "COLUNA_DATA", "COLUNA_DESASTRES",
                    "COLUNA_HORAS", "COLUNAS_COORDENADAS", "CONSTANTES_FIXAS"}

np.random.seed(SEMENTE)
keras.utils.set_random_seed(SEMENTE)


class Duplicador:
    """Escreve no terminal e, ao mesmo tempo, no arquivo da análise."""

    def __init__(self, terminal, arquivo):
        self.terminal = terminal
        self.arquivo = arquivo

    def write(self, texto):
        self.terminal.write(texto)
        self.arquivo.write(texto)
        self.arquivo.flush()

    def flush(self):
        self.terminal.flush()
        self.arquivo.flush()

    def isatty(self):
        return self.terminal.isatty()


def iniciar_registro():
    """Passa a copiar tudo o que aparece no terminal para um arquivo novo."""
    os.makedirs(PASTA_ANALISES, exist_ok=True)
    agora = datetime.now()
    caminho = os.path.join(PASTA_ANALISES, f"analises_{agora:%Y-%m-%d_%H-%M-%S}.txt")
    arquivo = open(caminho, "w", encoding="utf-8")
    sys.stdout = Duplicador(sys.__stdout__, arquivo)
    sys.stderr = Duplicador(sys.__stderr__, arquivo)

    print(f"Análise iniciada em {agora:%d/%m/%Y %H:%M:%S}")
    print("Parâmetros usados:")
    for nome, valor in list(globals().items()):
        if nome.isupper() and nome not in CONSTANTES_FIXAS:
            print(f"  {nome} = {valor!r}")
    print("-" * 70)
    return caminho, arquivo


def colunas_obrigatorias():
    return set([COLUNA_MUNICIPIO, COLUNA_DATA, COLUNA_DESASTRES] + COLUNAS_COORDENADAS)


def colunas_necessarias():
    opcionais = COLUNAS_ENTRADA + [COLUNA_HORAS] + ([COLUNA_FASE_ENSO] if COLUNA_FASE_ENSO else [])
    return colunas_obrigatorias() | set(opcionais)


def entradas_disponiveis(df):
    """COLUNAS_ENTRADA que existem nos dados, na ordem da lista."""
    return [c for c in COLUNAS_ENTRADA if c in df.columns]


def detectar_formato_csv():
    """Descobre separador e codificação lendo só as primeiras linhas do arquivo."""
    # "utf-8-sig" remove o caractere invisível (BOM) que o Excel põe no início
    for codificacao in ("utf-8-sig", "latin-1"):
        for separador in dict.fromkeys([SEPARADOR_CSV, ";", "\t", ","]):
            try:
                amostra = pd.read_csv(CAMINHO_DADOS, sep=separador, nrows=5,
                                      encoding=codificacao)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            if amostra.shape[1] > 5:
                decimal = DECIMAL_CSV if separador != DECIMAL_CSV else "."
                return separador, decimal, codificacao, [c.strip() for c in amostra.columns]
    raise ValueError("Não foi possível identificar as colunas do arquivo. "
                     "Confira SEPARADOR_CSV e DECIMAL_CSV.")


def preparar_bloco(bloco):
    """Converte tipos e filtra um bloco do CSV logo após a leitura.

    Fazer isso bloco a bloco (e não no arquivo inteiro de uma vez) mantém o
    consumo de memória baixo: números viram float32 e textos são descartados.
    """
    bloco.columns = [c.strip() for c in bloco.columns]
    numericas = COLUNAS_ENTRADA + COLUNAS_COORDENADAS + [COLUNA_DESASTRES, COLUNA_HORAS]
    for col in numericas:
        if col not in bloco.columns:
            continue
        if bloco[col].dtype == object:
            bloco[col] = pd.to_numeric(bloco[col].astype(str).str.replace(",", "."),
                                       errors="coerce")
        bloco[col] = bloco[col].astype("float32")

    if not np.issubdtype(bloco[COLUNA_DATA].dtype, np.datetime64):
        datas = pd.to_datetime(bloco[COLUNA_DATA], format=FORMATO_DATA, errors="coerce")
        if datas.isna().mean() > 0.5:  # formato informado não bate com o arquivo
            datas = pd.to_datetime(bloco[COLUNA_DATA], dayfirst=True, errors="coerce")
        bloco[COLUNA_DATA] = datas
    bloco[COLUNA_MUNICIPIO] = bloco[COLUNA_MUNICIPIO].astype(str).str.strip()
    if COLUNA_FASE_ENSO and COLUNA_FASE_ENSO in bloco.columns:
        bloco[COLUNA_FASE_ENSO] = bloco[COLUNA_FASE_ENSO].astype("string").str.strip()
    bloco = bloco.dropna(subset=[COLUNA_DATA])

    if COLUNA_HORAS in bloco.columns and MINIMO_HORAS_ERA5 > 0:
        bloco = bloco[bloco[COLUNA_HORAS] >= MINIMO_HORAS_ERA5]
    return bloco


def ler_arquivo():
    extensao = os.path.splitext(CAMINHO_DADOS)[1].lower()
    if extensao in (".xlsx", ".xls"):
        blocos = [pd.read_excel(CAMINHO_DADOS)]
    else:
        separador, decimal, codificacao, colunas = detectar_formato_csv()
        faltando = colunas_obrigatorias() - set(colunas)
        if faltando:
            raise ValueError(f"Colunas obrigatórias não encontradas no arquivo: {sorted(faltando)}. "
                             f"Colunas existentes: {colunas}")
        ignoradas = [c for c in COLUNAS_ENTRADA if c not in colunas]
        if ignoradas:
            print(f"Aviso: colunas de entrada que não existem na planilha e serão ignoradas: {ignoradas}")
        necessarias = colunas_necessarias()
        blocos = pd.read_csv(CAMINHO_DADOS, sep=separador, decimal=decimal,
                             encoding=codificacao, chunksize=LINHAS_POR_BLOCO,
                             usecols=lambda c: c.strip() in necessarias)

    escolhidos = set(MUNICIPIOS_FILTRO) if MUNICIPIOS_FILTRO else None
    sorteador = np.random.RandomState(SEMENTE)
    partes, sorteados, linhas_lidas = [], {}, 0
    for bloco in blocos:
        linhas_lidas += len(bloco)
        bloco = preparar_bloco(bloco)
        if escolhidos is not None:
            bloco = bloco[bloco[COLUNA_MUNICIPIO].isin(escolhidos)]
        elif PERCENTUAL_DADOS < 100:
            for m in bloco[COLUNA_MUNICIPIO].unique():
                if m not in sorteados:
                    sorteados[m] = sorteador.rand() < PERCENTUAL_DADOS / 100
            bloco = bloco[bloco[COLUNA_MUNICIPIO].map(sorteados).astype(bool)]
        partes.append(bloco)
        print(f"  {linhas_lidas:,} linhas lidas", end="\r")
    print()
    df = pd.concat(partes, ignore_index=True)
    df[COLUNA_MUNICIPIO] = df[COLUNA_MUNICIPIO].astype("category")
    return df


def caminho_cache():
    """Nome do arquivo de cache, que muda quando muda algo que afeta a leitura."""
    if MUNICIPIOS_FILTRO:
        filtro = "filtro" + hashlib.md5("|".join(sorted(MUNICIPIOS_FILTRO)).encode()).hexdigest()[:8]
    else:
        filtro = f"pct{PERCENTUAL_DADOS}"
    modificado = int(os.path.getmtime(CAMINHO_DADOS))
    # As colunas lidas também entram no nome: mudar COLUNAS_ENTRADA refaz o cache
    colunas = hashlib.md5("|".join(sorted(colunas_necessarias())).encode()).hexdigest()[:6]
    nome = (f"cache_{os.path.basename(CAMINHO_DADOS)}_{modificado}_{filtro}"
            f"_h{MINIMO_HORAS_ERA5}_c{colunas}.pkl")
    return os.path.join(PASTA_SAIDA, nome)


def carregar_dados():
    if not os.path.exists(CAMINHO_DADOS):
        raise FileNotFoundError(f"Arquivo {CAMINHO_DADOS} não encontrado. "
                                "Confira o parâmetro CAMINHO_DADOS.")

    cache = caminho_cache()
    if USAR_CACHE and os.path.exists(cache):
        print(f"Usando dados já processados em {cache}")
        df = pd.read_pickle(cache)
    else:
        print(f"Lendo {CAMINHO_DADOS} ({os.path.getsize(CAMINHO_DADOS) / 1e6:.0f} MB) ...")
        df = ler_arquivo()

        df = (df.sort_values([COLUNA_MUNICIPIO, COLUNA_DATA])
                .drop_duplicates([COLUNA_MUNICIPIO, COLUNA_DATA])
                .reset_index(drop=True))
        df["municipio_id"] = df[COLUNA_MUNICIPIO].cat.remove_unused_categories() \
                                                 .cat.codes.astype("int32")

        entradas = entradas_disponiveis(df)
        df[entradas] = (df.groupby("municipio_id")[entradas]
                        .transform(lambda s: s.interpolate(limit_direction="both"))
                        .fillna(0).astype("float32"))
        if USAR_CACHE:
            df.to_pickle(cache)
            print(f"Dados processados salvos em {cache} (as próximas execuções serão mais rápidas)")

    df["evento"] = (df[COLUNA_DESASTRES].fillna(0) >= LIMIAR_DESASTRES).astype("float32")
    dias_com_registro = int(df["evento"].sum())
    marcar_inicio_eventos(df)

    memoria = df.memory_usage(deep=True).sum() / 1e6
    print(f"{df[COLUNA_MUNICIPIO].nunique()} municípios, {len(df):,} linhas "
          f"({memoria:.0f} MB em memória), "
          f"de {df[COLUNA_DATA].min():%d/%m/%Y} a {df[COLUNA_DATA].max():%d/%m/%Y}")
    print(f"Dias com registro de desastre (Desastres >= {LIMIAR_DESASTRES}): {dias_com_registro:,}")
    if APENAS_INICIO_EVENTO:
        print(f"Eventos distintos (início de cada evento): {int(df['evento'].sum()):,} "
              f"({100 * df['evento'].mean():.3f}% dos dias) | "
              f"dias de continuação: {int(df['continuacao'].sum()):,}")
    return df


def marcar_inicio_eventos(df):
    """Mantém como evento só o primeiro dia de cada sequência de registros.

    Um registro é o início de um evento quando não houve outro registro no mesmo
    município nos DIAS_ENTRE_EVENTOS dias anteriores. Os demais dias com
    registro são marcados na coluna "continuacao".
    """
    if not APENAS_INICIO_EVENTO:
        df["continuacao"] = np.float32(0)
        return

    grupos = df.groupby("municipio_id", sort=False)
    registro = df["evento"].values == 1
    teve_registro_recente = np.zeros(len(df), dtype=bool)
    for k in range(1, DIAS_ENTRE_EVENTOS + 1):
        registro_anterior = grupos["evento"].shift(k).fillna(0).values == 1
        distancia = (df[COLUNA_DATA] - grupos[COLUNA_DATA].shift(k)).dt.days.to_numpy(dtype=float)
        teve_registro_recente |= registro_anterior & (distancia <= DIAS_ENTRE_EVENTOS)

    df["evento"] = (registro & ~teve_registro_recente).astype("float32")
    df["continuacao"] = (registro & teve_registro_recente).astype("float32")


ESTACOES_ONI = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def limpar_caminho_digitado(texto):
    """Remove aspas e barras de escape que o terminal põe ao arrastar um arquivo."""
    return texto.strip().strip("'\"").replace("\\ ", " ").strip()


def baixar_oni():
    destino = os.path.join(PASTA_SAIDA, "oni_noaa_cpc.txt")
    print(f"Baixando o ONI de {URL_ONI_NOAA} ...")
    with urllib.request.urlopen(URL_ONI_NOAA, timeout=60) as resposta:
        conteudo = resposta.read()
    with open(destino, "wb") as arquivo:
        arquivo.write(conteudo)
    print(f"ONI salvo em {destino}")
    return destino


def obter_arquivo_oni():
    """Usa CAMINHO_ONI, ou pergunta no terminal onde está o arquivo do ONI."""
    if CAMINHO_ONI:
        caminho = os.path.expanduser(CAMINHO_ONI)
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo do ONI não encontrado: {caminho}. "
                                    "Confira o parâmetro CAMINHO_ONI.")
        return caminho

    if not sys.stdin.isatty():
        return baixar_oni()

    print("\n" + "=" * 70)
    print("Arquivo do ONI (NOAA/CPC)")
    print("Arraste o arquivo do ONI para esta janela e aperte Enter,")
    print("ou apenas aperte Enter para baixar direto da NOAA.")
    print("=" * 70)
    while True:
        caminho = os.path.expanduser(limpar_caminho_digitado(input("Arquivo do ONI: ")))
        if not caminho:
            try:
                return baixar_oni()
            except Exception as erro:
                print(f"Não foi possível baixar o ONI ({erro}). Informe o caminho do arquivo.")
                continue
        if os.path.isfile(caminho):
            return caminho
        print(f"Arquivo não encontrado: {caminho}. Tente de novo.")


def ler_oni(caminho):
    """Lê o ONI e devolve uma tabela com ano, mês e valor do ONI (°C).

    Aceita o oni.ascii.txt da CPC (cada trimestre, ex. DJF, é atribuído ao seu
    mês central) e o oni.data da NOAA PSL (uma linha por ano com 12 meses).
    """
    with open(caminho, encoding="utf-8", errors="ignore") as arquivo:
        linhas = arquivo.read().splitlines()

    registros = []
    for linha in linhas:  # formato CPC: SEAS YR TOTAL ANOM
        partes = linha.split()
        if len(partes) >= 4 and partes[0].upper() in ESTACOES_ONI and partes[1].isdigit():
            registros.append((int(partes[1]), ESTACOES_ONI[partes[0].upper()], float(partes[3])))

    if not registros:  # formato PSL: ano seguido de 12 valores mensais
        for linha in linhas:
            partes = linha.split()
            if len(partes) == 13 and partes[0].isdigit() and len(partes[0]) == 4:
                for mes, valor in enumerate(partes[1:], start=1):
                    valor = float(valor)
                    if valor > -99:  # -99.90 = mês sem dado
                        registros.append((int(partes[0]), mes, valor))

    if not registros:
        raise ValueError(f"Não foi possível ler o ONI em {caminho}. Use o oni.ascii.txt da "
                         "CPC ou o oni.data da NOAA PSL (veja o parâmetro CAMINHO_ONI).")

    oni = pd.DataFrame(registros, columns=["ano", "mes", "oni"]).drop_duplicates(["ano", "mes"])
    print(f"ONI lido de {caminho}: {len(oni)} meses, de {oni['mes'].iloc[0]:02d}/{oni['ano'].iloc[0]} "
          f"a {oni['mes'].iloc[-1]:02d}/{oni['ano'].iloc[-1]}")
    return oni


def adicionar_oni(df):
    """Junta o ONI a cada dia pelo mês, com as defasagens de DEFASAGENS_ONI."""
    oni = ler_oni(obter_arquivo_oni())

    # Índice mensal contínuo: mês = ano * 12 + (mês - 1)
    serie = pd.Series(oni["oni"].values, index=oni["ano"] * 12 + oni["mes"] - 1).sort_index()
    mes_dia = (df[COLUNA_DATA].dt.year * 12 + df[COLUNA_DATA].dt.month - 1).values
    primeiro = min(serie.index.min(), mes_dia.min() - max(DEFASAGENS_ONI, default=0))
    ultimo = mes_dia.max()
    sem_dado = int((np.arange(primeiro, ultimo + 1) > serie.index.max()).sum())
    # Meses depois do último ONI publicado repetem o último valor conhecido
    serie = serie.reindex(range(primeiro, ultimo + 1)).ffill()
    if sem_dado:
        print(f"Aviso: {sem_dado} meses no fim dos dados ainda sem ONI publicado; "
              "foi usado o último valor disponível.")

    colunas = []
    for k in DEFASAGENS_ONI:
        nome = "ONI" if k == 0 else f"ONI_lag{k}"
        df[nome] = serie.reindex(mes_dia - k).values.astype("float32")
        colunas.append(nome)
    df[colunas] = df[colunas].fillna(0)

    # Fase do próprio mês: só para a análise descritiva, nunca como entrada
    df["fase_ENSO_mes"] = fase_do_oni(serie.reindex(mes_dia).values)

    # Fase usada pelo modelo: a do ONI da menor defasagem (já conhecido no dia)
    if COLUNA_FASE_ENSO:
        base = serie.reindex(mes_dia - min(DEFASAGENS_ONI, default=1)).values
        df[COLUNA_FASE_ENSO] = fase_do_oni(base)
        dias = df.drop_duplicates(COLUNA_DATA)[COLUNA_FASE_ENSO].value_counts()
        print("Dias por fase do ENSO (fase usada pelo modelo): "
              + ", ".join(f"{fase} {n}" for fase, n in dias.items()))

    print(f"Colunas do ONI adicionadas: {colunas}")
    return df, colunas


FASES_ENSO = ["La Niña", "Neutro", "El Niño"]


def fase_do_oni(valores):
    valores = np.asarray(valores, dtype=float)
    return np.select([valores >= LIMIAR_FASE_ENSO, valores <= -LIMIAR_FASE_ENSO],
                     ["El Niño", "La Niña"], "Neutro")


def analisar_desastres_por_fase(df, fim_treino, fim_val):
    """Compara a frequência de desastres em El Niño, La Niña e Neutro."""
    from scipy.stats import chi2_contingency

    fase = df["fase_ENSO_mes"]
    tabela = df.groupby(fase).agg(
        meses=(COLUNA_DATA, lambda d: d.dt.to_period("M").nunique()),
        municipio_dias=("evento", "size"),
        dias_com_evento=("evento", "sum"),
        registros_desastre=(COLUNA_DESASTRES, "sum"),
    ).reindex(FASES_ENSO).dropna()
    tabela["taxa_evento_pct"] = 100 * tabela["dias_com_evento"] / tabela["municipio_dias"]
    tabela["registros_por_1000_dias"] = 1000 * tabela["registros_desastre"] / tabela["municipio_dias"]
    tabela["risco_relativo_vs_neutro"] = tabela["taxa_evento_pct"] / tabela.loc["Neutro", "taxa_evento_pct"]

    print(f"\n{'=' * 70}\nDESASTRES POR FASE DO ENSO (fase do próprio mês, período completo)\n{'=' * 70}")
    print(tabela.to_string(float_format=lambda v: f"{v:,.3f}"))

    chi2, p, gl, _ = chi2_contingency(pd.crosstab(fase, df["evento"]))
    print(f"\nTeste qui-quadrado (fase x dia com evento): chi2 = {chi2:,.1f}, gl = {gl}, p = {p:.3g}")
    print("p < 0,05 indica que a frequência de desastres difere entre as fases. Com milhões de"
          " linhas, até diferenças pequenas dão p baixo: olhe também o risco relativo.")

    # A mesma comparação dentro de cada período, porque os registros aumentaram
    # com o tempo e as fases não se distribuem igualmente entre os períodos
    periodo = np.select([df[COLUNA_DATA] <= fim_treino, df[COLUNA_DATA] <= fim_val],
                        ["1 treino", "2 validação"], "3 teste")
    por_periodo = (100 * df.groupby([periodo, fase])["evento"].mean()).unstack()
    por_periodo = por_periodo.reindex(columns=[f for f in FASES_ENSO if f in por_periodo.columns])
    print("\nTaxa de dias com evento (%) por período e fase:")
    print(por_periodo.to_string(float_format=lambda v: f"{v:.3f}"))

    tabela.to_csv(os.path.join(PASTA_SAIDA, "desastres_por_fase_enso.csv"))

    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    cores = {"La Niña": "#4C78A8", "Neutro": "#9E9E9E", "El Niño": "#E45756"}
    ax = eixos[0]
    ax.bar(tabela.index, tabela["taxa_evento_pct"], color=[cores[f] for f in tabela.index])
    for i, v in enumerate(tabela["taxa_evento_pct"]):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_title("Dias com desastre por fase do ENSO (período completo)")
    ax.set_ylabel("% dos município-dias")
    ax = eixos[1]
    largura = 0.25
    for j, f in enumerate(por_periodo.columns):
        ax.bar(np.arange(len(por_periodo)) + (j - 1) * largura, por_periodo[f], largura,
               label=f, color=cores[f])
    ax.set_xticks(np.arange(len(por_periodo)))
    ax.set_xticklabels([p[2:] for p in por_periodo.index])
    ax.set_title("Por período")
    ax.set_ylabel("% dos município-dias")
    ax.legend()
    plt.tight_layout()
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminho = os.path.join(PASTA_GRAFICOS, "analises_desastres_por_fase_enso.png")
    plt.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"Tabela salva em {PASTA_SAIDA}/desastres_por_fase_enso.csv e gráfico em {caminho}")


def adicionar_risco_historico_fase(df, fim_treino):
    """Risco relativo de desastre de cada município em cada fase, medido no treino.

    risco = taxa suavizada do município na fase / taxa média do município.
    Usa só o período de treino, para não vazar informação do futuro.
    """
    treino = df[COLUNA_DATA] <= fim_treino
    taxa_global = df.loc[treino, "evento"].mean()
    por_municipio = df[treino].groupby("municipio_id")["evento"].mean()
    taxa_municipio = df["municipio_id"].map(por_municipio).fillna(taxa_global).values

    grupos = df[treino].groupby(["municipio_id", COLUNA_FASE_ENSO])["evento"].agg(["sum", "count"])
    chave = pd.MultiIndex.from_arrays([df["municipio_id"], df[COLUNA_FASE_ENSO]])
    eventos = grupos["sum"].reindex(chave).fillna(0).values
    dias = grupos["count"].reindex(chave).fillna(0).values

    a = SUAVIZACAO_RISCO_FASE
    taxa_fase = (eventos + a * taxa_municipio) / (dias + a)
    # Município sem nenhum desastre no treino: sem histórico, risco neutro (1)
    risco = np.where(taxa_municipio > 0, taxa_fase / np.maximum(taxa_municipio, 1e-9), 1.0)
    df["risco_fase_municipio"] = risco.astype("float32")

    resumo = df[treino].groupby(COLUNA_FASE_ENSO)["risco_fase_municipio"].mean().reindex(FASES_ENSO)
    print("Risco histórico médio por fase (treino; 1 = média do município): "
          + ", ".join(f"{f} {v:.2f}" for f, v in resumo.dropna().items()))
    return df, "risco_fase_municipio"


def adicionar_variaveis_chuva(df, fim_treino):
    """Cria chuva acumulada de prazo longo e o percentil da chuva no município."""
    colunas = []
    if "precipitacao" not in df.columns:
        return df, colunas

    grupos = df.groupby("municipio_id", sort=False)["precipitacao"]
    for dias in CHUVA_ACUMULADA_LONGA:
        nome = f"chuva_acum_{dias}d"
        df[nome] = grupos.transform(lambda s: s.rolling(dias, min_periods=1).sum()).astype("float32")
        colunas.append(nome)

    treino = (df[COLUNA_DATA] <= fim_treino).values
    indices = df.groupby("municipio_id", sort=False).indices
    for col in [c for c in COLUNAS_PERCENTIL_MUNICIPIO if c in df.columns]:
        valores = df[col].values
        percentil = np.full(len(df), 0.5, dtype="float32")
        for idx in indices.values():
            referencia = np.sort(valores[idx][treino[idx]])
            if len(referencia):
                percentil[idx] = np.searchsorted(referencia, valores[idx], side="right") / len(referencia)
        nome = f"{col}_pct_mun"
        df[nome] = percentil
        colunas.append(nome)

    print(f"Variáveis de chuva adicionadas: {colunas}")
    return df, colunas


def adicionar_variaveis_hidrologicas(df, fim_treino):
    """API (umidade antecedente), dias secos consecutivos e SPI simplificado."""
    from scipy.signal import lfilter

    colunas = []
    if "precipitacao" not in df.columns:
        return df, colunas
    chuva = df["precipitacao"].fillna(0).values.astype("float64")
    indices = df.groupby("municipio_id", sort=False).indices

    api = np.zeros(len(df), dtype="float32")
    secos = np.zeros(len(df), dtype="float32")
    for idx in indices.values():
        p = chuva[idx]
        api[idx] = lfilter([1.0], [1.0, -K_API], p)  # API(t) = K x API(t-1) + P(t)
        seco = p < LIMIAR_DIA_SECO_MM
        acumulado = np.cumsum(seco)
        ultimo_dia_chuvoso = np.maximum.accumulate(np.where(~seco, acumulado, 0))
        secos[idx] = acumulado - ultimo_dia_chuvoso
    df["api"] = api
    df["dias_secos_consecutivos"] = secos
    colunas += ["api", "dias_secos_consecutivos"]

    grupos = df.groupby("municipio_id", sort=False)["precipitacao"]
    mes = df[COLUNA_DATA].dt.month.values
    treino = (df[COLUNA_DATA] <= fim_treino).values
    for janela in JANELAS_SPI:
        nome_acum = f"chuva_acum_{janela}d"
        if nome_acum not in df.columns:
            df[nome_acum] = grupos.transform(lambda s: s.rolling(janela, min_periods=1).sum()).astype("float32")
        valor = np.log1p(df[nome_acum].clip(lower=0).values)
        referencia = pd.DataFrame({"m": df["municipio_id"].values[treino], "mes": mes[treino],
                                   "v": valor[treino]}).groupby(["m", "mes"])["v"].agg(["mean", "std"])
        chave = pd.MultiIndex.from_arrays([df["municipio_id"].values, mes])
        media = referencia["mean"].reindex(chave).values
        desvio = referencia["std"].reindex(chave).values
        spi = (valor - media) / np.where(desvio > 0, desvio, np.nan)
        df[f"spi_{janela}d"] = np.nan_to_num(spi, nan=0.0).clip(-5, 5).astype("float32")
        colunas.append(f"spi_{janela}d")

    print(f"Variáveis hidrológicas adicionadas: {colunas}")
    return df, colunas


def montar_matriz_entrada(df, fim_treino, colunas_extras=(), colunas_contexto=None):
    """Monta as matrizes de entrada (uma linha por município-dia), já normalizadas.

    Com colunas_contexto=None (arquitetura simples) tudo vai para a sequência e
    a matriz de contexto é None. Com uma lista, colunas_extras são as variáveis
    da sequência diária e o contexto recebe colunas_contexto mais fase do ENSO,
    coordenadas e época do ano.
    """
    if colunas_contexto is None:
        dinamicas = entradas_disponiveis(df) + list(colunas_extras)
    else:
        dinamicas = list(colunas_extras)
    entradas = df[dinamicas + list(colunas_contexto or [])].copy()
    for col in COLUNAS_LOG:
        if col in entradas:
            entradas[col] = np.log1p(entradas[col].clip(lower=0))
    if CRIAR_FASE_ENSO and COLUNA_FASE_ENSO and COLUNA_FASE_ENSO in df.columns:
        fases = pd.get_dummies(df[COLUNA_FASE_ENSO].astype(object).fillna("desconhecida"),
                               prefix="fase", dtype="float32")
        entradas = pd.concat([entradas, fases], axis=1)
    if USAR_COORDENADAS:
        entradas[COLUNAS_COORDENADAS] = df[COLUNAS_COORDENADAS]
    if USAR_SAZONALIDADE:
        dia_ano = df[COLUNA_DATA].dt.dayofyear
        entradas["sazon_sen"] = np.sin(2 * np.pi * dia_ano / 365.25)
        entradas["sazon_cos"] = np.cos(2 * np.pi * dia_ano / 365.25)

    scaler = StandardScaler() if TIPO_NORMALIZACAO == "standard" else MinMaxScaler()
    scaler.fit(entradas[df[COLUNA_DATA] <= fim_treino].values)
    X = scaler.transform(entradas.values).astype("float32")
    nomes = list(entradas.columns)
    if colunas_contexto is None:
        return X, None, nomes, []
    n = len(dinamicas)
    return X[:, :n], X[:, n:], nomes[:n], nomes[n:]


def calcular_janelas_validas(df):
    """Retorna o índice da última linha de cada janela válida e o alvo de cada linha.

    Uma janela é válida quando os TAMANHO_JANELA dias são consecutivos, do mesmo
    município, e os dias futuros usados no alvo também existem. Com
    IGNORAR_CONTINUACAO_EVENTO, janelas cujo período previsto só tem dias de
    continuação de um evento (e nenhum início) também são descartadas.
    """
    grupos = df.groupby("municipio_id", sort=False)
    datas = df[COLUNA_DATA]

    inicio = grupos[COLUNA_DATA].shift(TAMANHO_JANELA - 1)
    valida = (datas - inicio).dt.days.eq(TAMANHO_JANELA - 1).values

    passos = range(1, HORIZONTE_PREVISAO + 1) if ALVO_JANELA_HORIZONTE else [HORIZONTE_PREVISAO]
    alvo = np.zeros(len(df), dtype="float32")
    continuacao = np.zeros(len(df), dtype="float32")
    for k in passos:
        data_futura = grupos[COLUNA_DATA].shift(-k)
        valida &= (data_futura - datas).dt.days.eq(k).values
        alvo = np.maximum(alvo, grupos["evento"].shift(-k).fillna(0).values)
        continuacao = np.maximum(continuacao, grupos["continuacao"].shift(-k).fillna(0).values)

    if APENAS_INICIO_EVENTO and IGNORAR_CONTINUACAO_EVENTO:
        descartadas = valida & (alvo == 0) & (continuacao == 1)
        valida &= ~descartadas
        print(f"Janelas descartadas por preverem só dias de continuação: {int(descartadas.sum()):,}")

    return np.where(valida)[0], alvo


def alvo_chuva_extrema(df, finais, fim_treino):
    """Alvo = chuva dos próximos HORIZONTE_PREVISAO dias acima do limiar de chuva extrema."""
    grupos = df.groupby("municipio_id", sort=False)["precipitacao"]
    chuva_futura = np.zeros(len(df), dtype="float32")
    for k in range(1, HORIZONTE_PREVISAO + 1):
        chuva_futura += grupos.shift(-k).fillna(0).values.astype("float32")

    municipio = df["municipio_id"].values
    if TIPO_LIMIAR_CHUVA == "absoluto":
        limiar = np.full(len(df), LIMIAR_CHUVA_MM, dtype="float32")
        descricao = f"{LIMIAR_CHUVA_MM:g} mm em {HORIZONTE_PREVISAO} dias (todos os municípios)"
    else:
        treino = finais[df[COLUNA_DATA].values[finais] <= np.datetime64(fim_treino)]
        por_municipio = pd.Series(chuva_futura[treino]).groupby(municipio[treino]) \
            .quantile(PERCENTIL_CHUVA_EXTREMA)
        geral = np.quantile(chuva_futura[treino], PERCENTIL_CHUVA_EXTREMA)
        limiar = pd.Series(municipio).map(por_municipio).fillna(geral).values.astype("float32")
        descricao = (f"percentil {PERCENTIL_CHUVA_EXTREMA:.0%} do município: mediana de "
                     f"{np.median(por_municipio):.1f} mm em {HORIZONTE_PREVISAO} dias "
                     f"(de {por_municipio.min():.1f} a {por_municipio.max():.1f} mm)")

    alvo = ((chuva_futura >= limiar) & (chuva_futura > 0)).astype("float32")
    print(f"Alvo = chuva extrema: {descricao}")
    print(f"Dias com chuva extrema nos próximos {HORIZONTE_PREVISAO} dias: "
          f"{100 * alvo[finais].mean():.2f}% das amostras")
    return alvo


def avaliar_associacao_desastres(previsoes):
    """Mede quantos desastres acontecem nos dias com chuva extrema (prevista e real)."""
    from scipy.stats import chi2_contingency

    print(f"\n{'=' * 70}\nASSOCIAÇÃO COM DESASTRES (tag de desastre, período de teste)\n{'=' * 70}")
    desastre = previsoes["desastre_real"].values
    linhas = []
    for nome, marcado in [("alerta de chuva extrema (previsto)", previsoes["alerta"].values == 1),
                          ("chuva extrema real", previsoes["real"].values == 1)]:
        taxa_com = desastre[marcado].mean() if marcado.any() else np.nan
        taxa_sem = desastre[~marcado].mean() if (~marcado).any() else np.nan
        _, p, _, _ = chi2_contingency(pd.crosstab(marcado, desastre))
        linhas.append({
            "grupo": nome,
            "dias_marcados_pct": 100 * marcado.mean(),
            "taxa_desastre_marcados_pct": 100 * taxa_com,
            "taxa_desastre_outros_pct": 100 * taxa_sem,
            "risco_relativo": taxa_com / taxa_sem if taxa_sem else np.nan,
            "desastres_nos_marcados_pct": 100 * desastre[marcado].sum() / max(desastre.sum(), 1),
            "p_qui_quadrado": p,
        })
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nrisco_relativo = quantas vezes a taxa de desastre nos dias marcados supera a dos outros dias.")
    print("desastres_nos_marcados_pct = parte dos desastres que caiu em dias marcados.")
    print("A linha 'chuva extrema real' mostra o máximo que um alerta de chuva perfeito alcançaria.")
    tabela.to_csv(os.path.join(PASTA_SAIDA, "associacao_chuva_desastre.csv"), index=False)


def climatologia_mensal(df, finais, alvo, fim_treino):
    """Probabilidade histórica do alvo por município e mês, calculada no treino."""
    datas = df[COLUNA_DATA].values[finais]
    treino = finais[datas <= np.datetime64(fim_treino)]
    historico = pd.DataFrame({
        "municipio_id": df["municipio_id"].values[treino],
        "mes": df[COLUNA_DATA].dt.month.values[treino],
        "evento": alvo[treino],
    })
    taxa_municipio = historico.groupby("municipio_id")["evento"].mean()
    tabela = historico.groupby(["municipio_id", "mes"])["evento"] \
        .agg(dias="size", eventos="sum").reset_index()
    tabela["taxa_municipio"] = tabela["municipio_id"].map(taxa_municipio)

    a = SUAVIZACAO_CLIMATOLOGIA
    tabela["probabilidade"] = (tabela["eventos"] + a * tabela["taxa_municipio"]) / (tabela["dias"] + a)
    tabela["risco_relativo"] = np.where(tabela["taxa_municipio"] > 0,
                                        tabela["probabilidade"] / tabela["taxa_municipio"].clip(lower=1e-9),
                                        1.0)
    baixo, alto = CORTES_CATEGORIA_CLIMATOLOGIA
    tabela["categoria"] = np.select([tabela["risco_relativo"] < baixo, tabela["risco_relativo"] > alto],
                                    ["baixa", "alta"], "média")
    return tabela


def aplicar_climatologia(df, tabela):
    """Probabilidade, risco relativo e categoria climatológica de cada linha de df."""
    chave = pd.MultiIndex.from_arrays([df["municipio_id"].values, df[COLUNA_DATA].dt.month.values])
    indexada = tabela.set_index(["municipio_id", "mes"])
    probabilidade = indexada["probabilidade"].reindex(chave).values
    probabilidade = np.where(np.isnan(probabilidade), tabela["probabilidade"].mean(), probabilidade)
    risco = indexada["risco_relativo"].reindex(chave).fillna(1.0).values
    categoria = indexada["categoria"].reindex(chave).fillna("média").values
    return probabilidade.astype("float32"), risco.astype("float32"), categoria


def definir_datas_corte(df):
    if DATA_FIM_TREINO and DATA_FIM_VALIDACAO:
        return pd.Timestamp(DATA_FIM_TREINO), pd.Timestamp(DATA_FIM_VALIDACAO)
    datas = np.sort(df[COLUNA_DATA].unique())
    fim_treino = datas[int(len(datas) * PROPORCAO_TREINO) - 1]
    fim_val = datas[int(len(datas) * (PROPORCAO_TREINO + PROPORCAO_VALIDACAO)) - 1]
    return pd.Timestamp(fim_treino), pd.Timestamp(fim_val)


def criar_dataset(tensores, indices, alvo, pesos=None, embaralhar=False):
    """tensores = {"sequencia": X diário, "contexto": X de contexto ou None, "municipio": ids}."""
    deslocamentos = tf.range(-TAMANHO_JANELA + 1, 1)

    def montar_entradas(idx):
        entradas = {"sequencia": tf.gather(tensores["sequencia"], idx[:, None] + deslocamentos[None, :])}
        if tensores["contexto"] is not None:
            entradas["contexto"] = tf.gather(tensores["contexto"], idx)  # valores do último dia
        if USAR_EMBEDDING_MUNICIPIO:
            entradas["municipio"] = tf.gather(tensores["municipio"], idx)
        return entradas

    y = alvo[indices]
    if pesos is None:
        ds = tf.data.Dataset.from_tensor_slices(indices.astype("int32"))
    else:
        w = np.where(y == 1, pesos[1], pesos[0]).astype("float32")
        ds = tf.data.Dataset.from_tensor_slices((indices.astype("int32"), y, w))
    if embaralhar:
        ds = ds.shuffle(len(indices), seed=SEMENTE, reshuffle_each_iteration=True)
    ds = ds.batch(TAMANHO_LOTE)

    if pesos is None:
        ds = ds.map(montar_entradas, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda i, yb, wb: (montar_entradas(i), yb, wb),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def criar_dataset_validacao(tensores, indices, alvo):
    ds = criar_dataset(tensores, indices, alvo, pesos={0: 1.0, 1: 1.0})
    return ds.map(lambda x, y, w: (x, y))


def construir_modelo(n_variaveis, n_municipios, n_contexto=0):
    reg = keras.regularizers.l2(REGULARIZACAO_L2) if REGULARIZACAO_L2 > 0 else None
    entradas_modelo = {}

    # Ramo de contexto: características fixas da amostra (município, local,
    # época, ENSO, climatologia). Elas definem o estado inicial da LSTM, para a
    # rede ler a sequência de chuva já "sabendo" de que município e época se
    # trata (ideia da EA-LSTM), e voltam a ser usadas junto da saída da LSTM.
    estado_inicial, contexto = None, None
    if n_contexto:
        entrada_ctx = layers.Input(shape=(n_contexto,), name="contexto")
        entradas_modelo["contexto"] = entrada_ctx
        contexto = layers.Dense(UNIDADES_CONTEXTO, activation="relu", kernel_regularizer=reg)(entrada_ctx)
        if not BIDIRECIONAL:
            estado_inicial = [layers.Dense(UNIDADES_LSTM[0], activation="tanh")(contexto),
                              layers.Dense(UNIDADES_LSTM[0], activation="tanh")(contexto)]

    entrada_seq = layers.Input(shape=(TAMANHO_JANELA, n_variaveis), name="sequencia")
    entradas_modelo["sequencia"] = entrada_seq
    x = entrada_seq
    for i, unidades in enumerate(UNIDADES_LSTM):
        retorna_sequencia = i < len(UNIDADES_LSTM) - 1
        lstm = layers.LSTM(unidades, return_sequences=retorna_sequencia,
                           recurrent_dropout=DROPOUT_RECORRENTE, kernel_regularizer=reg)
        if BIDIRECIONAL:
            x = layers.Bidirectional(lstm)(x)
        elif i == 0 and estado_inicial is not None:
            x = lstm(x, initial_state=estado_inicial)
        else:
            x = lstm(x)
        x = layers.Dropout(DROPOUT)(x)

    juntar = [x]
    if contexto is not None:
        juntar.append(contexto)
    if USAR_EMBEDDING_MUNICIPIO:
        entrada_mun = layers.Input(shape=(), dtype="int32", name="municipio")
        entradas_modelo["municipio"] = entrada_mun
        juntar.append(layers.Embedding(n_municipios, DIMENSAO_EMBEDDING,
                                       embeddings_regularizer=reg)(entrada_mun))
    if len(juntar) > 1:
        x = layers.Concatenate()(juntar)

    if UNIDADES_DENSA > 0:
        x = layers.Dense(UNIDADES_DENSA, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(DROPOUT)(x)
    saida = layers.Dense(1, activation="sigmoid")(x)

    modelo = keras.Model(entradas_modelo, saida)
    if FUNCAO_PERDA == "focal":
        perda = keras.losses.BinaryFocalCrossentropy(gamma=GAMMA_FOCAL)
    else:
        perda = "binary_crossentropy"
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=TAXA_APRENDIZADO),
        loss=perda,
        metrics=[keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.Precision(name="precisao"),
                 keras.metrics.Recall(name="recall")],
    )
    return modelo

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def criar_calibrador(y_val, p_val):
    if not CALIBRAR_PROBABILIDADES or len(np.unique(y_val)) < 2:
        return lambda p: p
    regressao = LogisticRegression().fit(logit(p_val)[:, None], y_val)
    return lambda p: regressao.predict_proba(logit(p)[:, None])[:, 1]


def escolher_limiar(y, p, configuracao, criterio):
    if configuracao != "auto":
        return float(configuracao)
    if len(np.unique(y)) < 2:
        return 0.5

    if criterio in ("youden", "equilibrio"):
        fpr, tpr, limiares = roc_curve(y, p)
        if criterio == "youden":
            pontuacao = tpr - fpr
        else:  # maior valor possível para a PIOR das duas taxas de acerto
            pontuacao = np.minimum(tpr, 1 - fpr)
        pontuacao[~np.isfinite(limiares)] = -np.inf  # o primeiro limiar do roc_curve é infinito
        return float(limiares[np.argmax(pontuacao)])

    precisao, recall, limiares = precision_recall_curve(y, p)
    precisao, recall = precisao[:-1], recall[:-1]  # alinha com os limiares
    if criterio == "recall_minimo":
        atende = recall >= RECALL_MINIMO
        if not atende.any():
            return float(limiares[0])
        return float(limiares[np.argmax(np.where(atende, precisao, -1))])

    if criterio != "f1":
        raise ValueError(f"Critério de limiar desconhecido: {criterio!r}. "
                         "Use 'f1', 'youden' ou 'recall_minimo'.")
    f1 = 2 * precisao * recall / (precisao + recall + 1e-12)
    return float(limiares[np.argmax(f1)])


def imprimir_metricas(titulo, y, p, limiar):
    pred = (p >= limiar).astype(int)
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
    imprimir_metricas_balanceadas(y, pred)
    if np.ndim(limiar):
        texto_limiar = (f"por município (mediana {np.median(limiar):.4f}, "
                        f"de {np.min(limiar):.4f} a {np.max(limiar):.4f})")
    else:
        texto_limiar = f"{limiar:.4f}"
    print(f"Amostras: {len(y)} | eventos: {int(y.sum())} | limiar: {texto_limiar}")
    if len(np.unique(y)) > 1:
        print(f"AUC-ROC: {roc_auc_score(y, p):.4f}")
        print(f"PR-AUC:  {average_precision_score(y, p):.4f} "
              f"(um modelo aleatório teria {y.mean():.4f})")
    print("Matriz de confusão [[VN, FP], [FN, VP]]:")
    print(confusion_matrix(y, pred, labels=[0, 1]))
    print(classification_report(y, pred, labels=[0, 1], zero_division=0,
                                target_names=["sem evento", "evento"]))


def imprimir_metricas_balanceadas(y, pred):
    """Sensibilidade, especificidade, acurácia balanceada e média G.

    Métricas indicadas para classes desbalanceadas (He & Garcia, 2009):
    acurácia balanceada = média das duas taxas de acerto (Brodersen et al., 2010);
    média G = média geométrica das duas (Kubat & Matwin, 1997).
    """
    sensibilidade = pred[y == 1].mean() if (y == 1).any() else np.nan
    especificidade = (1 - pred[y == 0]).mean() if (y == 0).any() else np.nan
    print(f"Sensibilidade (eventos detectados): {100 * sensibilidade:.1f}% | "
          f"Especificidade (dias sem evento reconhecidos): {100 * especificidade:.1f}%")
    print(f"Acurácia balanceada: {100 * (sensibilidade + especificidade) / 2:.1f}% | "
          f"Média G: {100 * np.sqrt(sensibilidade * especificidade):.1f}% | "
          f"Meta de 50% em cada classe: "
          f"{'ATINGIDA' if min(sensibilidade, especificidade) >= 0.5 else 'não atingida'}")


def limiares_por_municipio(prev_val, limiar_geral):
    """Limiar próprio para cada município, escolhido na validação.

    Devolve o dicionário de limiares próprios e o limiar dos demais municípios.
    Esse limiar dos demais é escolhido só com os municípios SEM limiar próprio:
    eles costumam ter probabilidades mais baixas que os grandes, e um limiar
    escolhido com todos juntos ficaria alto demais para eles (quase não alertavam).
    """
    limiares = {}
    if not LIMIAR_POR_MUNICIPIO:
        return limiares, limiar_geral
    for municipio, g in prev_val.groupby("municipio", observed=True):
        y = g["real"].values
        if y.sum() >= MINIMO_EVENTOS_LIMIAR_MUNICIPIO and len(np.unique(y)) == 2:
            limiares[municipio] = escolher_limiar(y, g["probabilidade"].values, "auto",
                                                  CRITERIO_LIMIAR_MUNICIPIO)
    demais = prev_val[~prev_val["municipio"].isin(list(limiares))]
    limiar_demais = limiar_geral
    if len(demais) and demais["real"].nunique() == 2:
        limiar_demais = escolher_limiar(demais["real"].values, demais["probabilidade"].values, "auto",
                                        CRITERIO_LIMIAR_MUNICIPIO)
    print(f"\nLimiar próprio para {len(limiares)} municípios (com {MINIMO_EVENTOS_LIMIAR_MUNICIPIO}+ "
          f"eventos na validação); os demais usam um limiar comum de {limiar_demais:.4f}, escolhido "
          f"só com eles (o limiar com todos juntos seria {limiar_geral:.4f})")
    return limiares, limiar_demais


def metricas_por_municipio(previsoes):
    linhas = []
    for municipio, g in previsoes.groupby("municipio", observed=True):
        y, p = g["real"].values, g["probabilidade"].values
        pred = g["pontuacao"].values >= g["limiar"].values
        vp = int((pred & (y == 1)).sum())
        linhas.append({
            "municipio": municipio,
            "limiar": g["limiar"].iloc[0],
            "eventos_reais": int(y.sum()),
            "alertas_emitidos": int(pred.sum()),
            "eventos_detectados": vp,
            "recall": vp / y.sum() if y.sum() else np.nan,
            "precisao": vp / pred.sum() if pred.sum() else np.nan,
            "auc_roc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "probabilidade_media": p.mean(),
        })
    return pd.DataFrame(linhas).sort_values("eventos_reais", ascending=False)


def probabilidade_pelo_menos(probabilidades, n):
    """P(pelo menos n municípios atingidos), tratando os municípios como independentes.

    Calcula a distribuição do número de municípios atingidos (distribuição
    binomial de Poisson), acompanhando só as contagens de 0 a n-1.
    """
    distribuicao = np.zeros(n)
    distribuicao[0] = 1.0
    for p in probabilidades:
        distribuicao[1:] = distribuicao[1:] * (1 - p) + distribuicao[:-1] * p
        distribuicao[0] *= 1 - p
    return 1 - distribuicao.sum()


def agregar_estado(previsoes):
    """Agrega as probabilidades municipais em uma previsão diária para SC."""
    n = MINIMO_MUNICIPIOS_EVENTO_ESTADUAL
    grupos = previsoes.groupby("data")
    estado = grupos.agg(
        municipios_esperados=("probabilidade", "sum"),
        municipios_atingidos=("real", "sum"),
    )
    estado["probabilidade_estado"] = grupos["probabilidade"].apply(
        lambda s: probabilidade_pelo_menos(s.values, n))
    estado["evento_estado"] = (estado["municipios_atingidos"] >= n).astype(int)
    return estado.reset_index()


def percentil_recente_municipal(prev_val, prev_teste):
    """Posição de cada previsão municipal entre as previsões do último ano.

    Para cada dia, a referência são todas as previsões (de todos os municípios)
    dos JANELA_LIMIAR_MUNICIPIO_DIAS dias anteriores. Só previsões passadas
    entram na referência, nunca o resultado real.
    """
    juntos = pd.concat([prev_val.assign(_parte="val"), prev_teste.assign(_parte="teste")])
    juntos = juntos.sort_values("data", kind="stable").reset_index(drop=True)
    datas = pd.to_datetime(juntos["data"]).values
    previsoes = juntos["probabilidade"].values
    janela = np.timedelta64(JANELA_LIMIAR_MUNICIPIO_DIAS, "D")

    percentil = np.full(len(juntos), 0.5)
    dias_historico = np.zeros(len(juntos), dtype=int)
    dias, inicio_dia = np.unique(datas, return_index=True)
    fim_dia = np.append(inicio_dia[1:], len(juntos))
    for dia, ini, fim in zip(dias, inicio_dia, fim_dia):
        inicio_ref = np.searchsorted(datas, dia - janela, side="left")
        referencia = np.sort(previsoes[inicio_ref:ini])
        if len(referencia):
            percentil[ini:fim] = np.searchsorted(referencia, previsoes[ini:fim], side="left") / len(referencia)
        dias_historico[ini:fim] = (dia - datas[inicio_ref]) // np.timedelta64(1, "D") if len(referencia) else 0
    juntos["percentil_recente"] = percentil
    juntos["dias_historico"] = dias_historico

    val = juntos[juntos["_parte"] == "val"].drop(columns="_parte").reset_index(drop=True)
    teste = juntos[juntos["_parte"] == "teste"].drop(columns="_parte").reset_index(drop=True)
    return val, teste


def tabela_niveis_risco(y, percentil, titulo):
    """Taxa de evento em cada nível de risco (pela posição entre as previsões recentes)."""
    nivel = np.digitize(percentil, CORTES_NIVEIS_RISCO)
    taxa_media = y.mean()
    linhas = []
    for i, nome in enumerate(NOMES_NIVEIS_RISCO):
        marcado = nivel == i
        taxa = y[marcado].mean() if marcado.any() else np.nan
        linhas.append({"nivel": nome, "dias_pct": 100 * marcado.mean(), "taxa_evento_pct": 100 * taxa,
                       "risco_vs_media": taxa / taxa_media if taxa_media else np.nan,
                       "eventos_pct": 100 * y[marcado].sum() / max(y.sum(), 1)})
    tabela = pd.DataFrame(linhas)
    tabela.insert(1, "comunicado", tabela["nivel"].map(COMUNICADO_POR_NIVEL))
    print(f"\nNíveis de risco - {titulo} (cortes nos percentis {CORTES_NIVEIS_RISCO} das previsões recentes):")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("taxa_evento_pct = chance real de evento nos dias daquele nível; eventos_pct = parte dos eventos "
          "que caiu nele.")

    nomes_nivel = np.array(NOMES_NIVEIS_RISCO, dtype=object)[nivel]
    comunicado = pd.Series(nomes_nivel).map(COMUNICADO_POR_NIVEL).values
    linhas = []
    for nome in ORDEM_COMUNICADOS:
        marcado = comunicado == nome
        taxa = y[marcado].mean() if marcado.any() else np.nan
        linhas.append({"comunicado": nome, "dias_pct": 100 * marcado.mean(), "taxa_evento_pct": 100 * taxa,
                       "risco_vs_media": taxa / taxa_media if taxa_media else np.nan,
                       "eventos_pct": 100 * y[marcado].sum() / max(y.sum(), 1)})
    print(f"\nComunicados - {titulo}:")
    print(pd.DataFrame(linhas).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return tabela, nomes_nivel, comunicado


def percentil_recente_estado(estado_val, estado_teste):
    """Posição da previsão de cada dia entre as previsões dos dias anteriores.

    Para cada dia, calcula a fração das previsões dos JANELA_LIMIAR_ESTADO_DIAS
    dias anteriores que ficaram abaixo da previsão do dia (0 = a menor do
    período, 1 = a maior). Usa só previsões passadas, nunca o resultado real.
    """
    juntos = pd.concat([estado_val.assign(_parte="val"), estado_teste.assign(_parte="teste")])
    juntos = juntos.sort_values("data").reset_index(drop=True)
    datas = pd.to_datetime(juntos["data"]).values
    previsoes = juntos["probabilidade_estado"].values
    janela = np.timedelta64(JANELA_LIMIAR_ESTADO_DIAS, "D")

    percentil = np.full(len(juntos), np.nan)
    historico = np.zeros(len(juntos), dtype=int)
    inicio = 0
    for i in range(len(juntos)):
        while datas[i] - datas[inicio] > janela:
            inicio += 1
        anteriores = previsoes[inicio:i]
        historico[i] = len(anteriores)
        if len(anteriores):
            percentil[i] = np.mean(anteriores < previsoes[i])
    juntos["percentil_recente"] = np.nan_to_num(percentil, nan=0.5)
    juntos["dias_historico"] = historico

    val = juntos[juntos["_parte"] == "val"].drop(columns="_parte").reset_index(drop=True)
    teste = juntos[juntos["_parte"] == "teste"].drop(columns="_parte").reset_index(drop=True)
    return val, teste


def salvar_graficos(historico, previsoes_teste, estado_teste, limiar_estado):
    fig, eixos = plt.subplots(2, 2, figsize=(15, 9))

    ax = eixos[0, 0]
    ax.plot(historico.history["loss"], label="treino")
    ax.plot(historico.history["val_loss"], label="validação")
    ax.set_title("Perda (loss)")
    ax.set_xlabel("época")
    ax.legend()

    ax = eixos[0, 1]
    ax.plot(historico.history["pr_auc"], label="treino")
    ax.plot(historico.history["val_pr_auc"], label="validação")
    ax.set_title("PR-AUC (quanto maior, melhor)")
    ax.set_xlabel("época")
    ax.legend()

    ax = eixos[1, 0]
    y, p = previsoes_teste["real"].values, previsoes_teste["probabilidade"].values
    if len(np.unique(y)) > 1:
        precisao, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precisao)
        ax.axhline(y.mean(), color="gray", linestyle="--", label="modelo aleatório")
    ax.set_title("Curva precisão-recall (municípios, teste)")
    ax.set_xlabel("recall")
    ax.set_ylabel("precisão")
    ax.legend()

    ax = eixos[1, 1]
    ax.plot(estado_teste["data"], estado_teste["probabilidade_estado"],
            linewidth=0.7, label="probabilidade prevista")
    dias_evento = estado_teste[estado_teste["evento_estado"] == 1]
    ax.scatter(dias_evento["data"], np.ones(len(dias_evento)), color="red", s=6,
               label=f"dia com {MINIMO_MUNICIPIOS_EVENTO_ESTADUAL}+ municípios atingidos")
    if limiar_estado is not None:
        ax.axhline(limiar_estado, color="gray", linestyle="--", label="limiar")
    else:
        alertas = estado_teste[estado_teste["alerta"] == 1]
        ax.scatter(alertas["data"], np.full(len(alertas), 0.95), color="gray", s=4,
                   label="dia com alerta (limiar adaptativo)")
    ax.set_title("Previsão estadual (teste)")
    ax.legend(loc="lower left")

    plt.tight_layout()
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminho = os.path.join(PASTA_GRAFICOS, "00_treinamento_e_previsao_estadual.png")
    plt.savefig(caminho, dpi=120)
    print(f"Gráficos salvos em {caminho}")

def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    df = carregar_dados()

    fim_treino, fim_val = definir_datas_corte(df)
    print(f"Treino até {fim_treino:%d/%m/%Y} | validação até {fim_val:%d/%m/%Y} | teste depois")

    # Variáveis da sequência diária (dinâmicas) e de contexto (fixas na amostra)
    df, extras_dinamicas = adicionar_variaveis_chuva(df, fim_treino)
    df, colunas_hidro = adicionar_variaveis_hidrologicas(df, fim_treino)
    extras_dinamicas += [c for c in colunas_hidro if c not in extras_dinamicas]
    extras_contexto = []
    if USAR_ONI:
        df, colunas_oni = adicionar_oni(df)
        extras_contexto += colunas_oni
        if ANALISAR_DESASTRES_POR_FASE:
            analisar_desastres_por_fase(df, fim_treino, fim_val)
        if USAR_RISCO_HISTORICO_FASE:
            df, coluna_risco = adicionar_risco_historico_fase(df, fim_treino)
            extras_contexto.append(coluna_risco)

    finais, alvo_desastre = calcular_janelas_validas(df)
    if ALVO == "chuva_extrema":
        alvo = alvo_chuva_extrema(df, finais, fim_treino)
        nome_evento = "chuva extrema"
    elif ALVO == "desastre":
        alvo = alvo_desastre
        nome_evento = "desastre"
    else:
        raise ValueError(f"ALVO desconhecido: {ALVO!r}. Use 'desastre' ou 'chuva_extrema'.")

    if USAR_CLIMATOLOGIA:
        clima = climatologia_mensal(df, finais, alvo, fim_treino)
        df["clima_prob"], df["clima_risco"], _ = aplicar_climatologia(df, clima)
        extras_contexto += ["clima_prob", "clima_risco"]
        por_mes = clima.groupby("mes")["risco_relativo"].mean()
        print(f"Climatologia de {nome_evento} (risco relativo médio por mês, treino): "
              + ", ".join(f"{mes}: {r:.2f}" for mes, r in por_mes.items()))

    if ARQUITETURA == "separada":
        grupos = GRUPO_GATILHO + GRUPO_UMIDADE + GRUPO_SECA + GRUPO_ATMOSFERA
        dinamicas = [c for c in dict.fromkeys(grupos) if c in df.columns]
        X, X_contexto, nomes_variaveis, nomes_contexto = montar_matriz_entrada(
            df, fim_treino, dinamicas, extras_contexto)
    elif ARQUITETURA == "simples":
        X, X_contexto, nomes_variaveis, nomes_contexto = montar_matriz_entrada(
            df, fim_treino, extras_dinamicas + extras_contexto)
    else:
        raise ValueError(f"ARQUITETURA desconhecida: {ARQUITETURA!r}. Use 'separada' ou 'simples'.")
    print(f"Variáveis da sequência diária ({len(nomes_variaveis)}): {nomes_variaveis}")
    if nomes_contexto:
        print(f"Variáveis de contexto ({len(nomes_contexto)}): {nomes_contexto}")

    datas_finais = df[COLUNA_DATA].values[finais]
    idx_treino = finais[datas_finais <= np.datetime64(fim_treino)]
    idx_val = finais[(datas_finais > np.datetime64(fim_treino)) & (datas_finais <= np.datetime64(fim_val))]
    idx_teste = finais[datas_finais > np.datetime64(fim_val)]

    # Mantém todos os eventos e uma fração dos dias sem evento no treino
    positivos = idx_treino[alvo[idx_treino] == 1]
    negativos = idx_treino[alvo[idx_treino] == 0]
    n_neg = int(len(negativos) * FRACAO_NEGATIVOS_TREINO)
    negativos = np.random.choice(negativos, n_neg, replace=False)
    idx_treino_amostrado = np.concatenate([positivos, negativos])

    for nome, idx in [("treino (amostrado)", idx_treino_amostrado),
                      ("validação", idx_val), ("teste", idx_teste)]:
        print(f"{nome:>20}: {len(idx):>9} amostras, {int(alvo[idx].sum()):>6} com evento")

    pesos = {0: 1.0, 1: 1.0}
    if USAR_PESO_CLASSES and len(positivos):
        n = len(idx_treino_amostrado)
        pesos = {0: n / (2 * len(negativos)), 1: n / (2 * len(positivos))}

    tensores = {
        "sequencia": tf.constant(np.ascontiguousarray(X)),
        "contexto": tf.constant(np.ascontiguousarray(X_contexto)) if X_contexto is not None else None,
        "municipio": tf.constant(df["municipio_id"].values),
    }
    del X, X_contexto
    ds_treino = criar_dataset(tensores, idx_treino_amostrado, alvo, pesos, embaralhar=True)
    ds_val = criar_dataset_validacao(tensores, idx_val, alvo)

    n_contexto = int(tensores["contexto"].shape[1]) if tensores["contexto"] is not None else 0
    modelo = construir_modelo(int(tensores["sequencia"].shape[1]), int(df["municipio_id"].max()) + 1,
                              n_contexto)
    modelo.summary()

    modo = "min" if METRICA_MONITORADA.endswith("loss") else "max"
    callbacks = [keras.callbacks.EarlyStopping(monitor=METRICA_MONITORADA, mode=modo,
                                               patience=PACIENCIA_EARLY_STOPPING,
                                               restore_best_weights=True)]
    if REDUZIR_TAXA_AUTOMATICAMENTE:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=METRICA_MONITORADA, mode=modo, factor=0.5,
            patience=max(1, PACIENCIA_EARLY_STOPPING // 2), min_lr=1e-5, verbose=1))
    historico = modelo.fit(
        ds_treino,
        validation_data=ds_val,
        epochs=EPOCAS,
        callbacks=callbacks,
        verbose=2,
    )

    def prever(idx):
        ds = criar_dataset(tensores, idx, alvo)
        return modelo.predict(ds, verbose=0).ravel()

    p_val_bruto, p_teste_bruto = prever(idx_val), prever(idx_teste)
    y_val, y_teste = alvo[idx_val], alvo[idx_teste]
    calibrar = criar_calibrador(y_val, p_val_bruto)
    p_val, p_teste = calibrar(p_val_bruto), calibrar(p_teste_bruto)

    def tabela_previsoes(idx, y, p):
        return pd.DataFrame({
            "municipio": df[COLUNA_MUNICIPIO].values[idx],
            "data": df[COLUNA_DATA].values[idx],
            "real": y.astype(int),
            "probabilidade": p,
            "desastre_real": alvo_desastre[idx].astype(int),
            "clima_prob": df["clima_prob"].values[idx] if USAR_CLIMATOLOGIA else np.nan,
        })

    prev_val = tabela_previsoes(idx_val, y_val, p_val)
    prev_teste = tabela_previsoes(idx_teste, y_teste, p_teste)

    # ---- Nível municipal ----
    if LIMIAR_MUNICIPIO_ADAPTATIVO:
        prev_val, prev_teste = percentil_recente_municipal(prev_val, prev_teste)
        y_teste, p_teste = prev_teste["real"].values, prev_teste["probabilidade"].values
        usados = prev_val["dias_historico"] >= MINIMO_HISTORICO_LIMIAR_ESTADO
        limiar_mun = escolher_limiar(prev_val.loc[usados, "real"].values,
                                     prev_val.loc[usados, "percentil_recente"].values,
                                     LIMIAR_MUNICIPIO, CRITERIO_LIMIAR_MUNICIPIO)
        prev_teste["pontuacao"] = prev_teste["percentil_recente"]
        prev_teste["limiar"] = limiar_mun
        print(f"\nLimiar municipal adaptativo: alerta quando a previsão supera {100 * limiar_mun:.1f}% das "
              f"previsões de todos os municípios nos {JANELA_LIMIAR_MUNICIPIO_DIAS} dias anteriores "
              f"(percentil escolhido na validação)")
        texto_limiar = f"limiar adaptativo por {CRITERIO_LIMIAR_MUNICIPIO}"
    else:
        limiar_mun = escolher_limiar(y_val, p_val, LIMIAR_MUNICIPIO, CRITERIO_LIMIAR_MUNICIPIO)
        limiares = {}
        if LIMIAR_MUNICIPIO == "auto":
            limiares, limiar_mun = limiares_por_municipio(prev_val, limiar_mun)
        prev_teste["pontuacao"] = prev_teste["probabilidade"]
        prev_teste["limiar"] = prev_teste["municipio"].map(limiares).astype(float).fillna(limiar_mun).values
        texto_limiar = (f"limiar por {CRITERIO_LIMIAR_MUNICIPIO}"
                        f"{', próprio por município' if limiares else ''}")
    imprimir_metricas(f"NÍVEL MUNICIPAL - teste (evento = {nome_evento}; cada linha = um município "
                      f"em um dia; {texto_limiar})",
                      y_teste, prev_teste["pontuacao"].values,
                      limiar_mun if LIMIAR_MUNICIPIO_ADAPTATIVO else prev_teste["limiar"].values)
    if LIMIAR_MUNICIPIO_ADAPTATIVO:
        _, prev_teste["nivel_risco"], prev_teste["comunicado"] = tabela_niveis_risco(
            y_teste, prev_teste["percentil_recente"].values, "municípios (teste)")
    if USAR_CLIMATOLOGIA and len(np.unique(y_teste)) == 2:
        auc_referencia = roc_auc_score(y_teste, prev_teste["clima_prob"].values)
        print(f"Referência - climatologia sozinha (risco do município no mês): AUC-ROC = {auc_referencia:.4f} "
              f"| LSTM: AUC-ROC = {roc_auc_score(y_teste, p_teste):.4f}")
    por_municipio = metricas_por_municipio(prev_teste)
    print("Municípios com mais eventos no teste:")
    print(por_municipio.head(15).to_string(index=False, float_format="%.3f"))

    # ---- Nível estadual ----
    estado_val, estado_teste = agregar_estado(prev_val), agregar_estado(prev_teste)
    if LIMIAR_ESTADO_ADAPTATIVO:
        estado_val, estado_teste = percentil_recente_estado(estado_val, estado_teste)
        coluna_estado = "percentil_recente"
        usados = estado_val["dias_historico"] >= MINIMO_HISTORICO_LIMIAR_ESTADO
        limiar_est = escolher_limiar(estado_val.loc[usados, "evento_estado"].values,
                                     estado_val.loc[usados, coluna_estado].values, LIMIAR_ESTADO,
                                     CRITERIO_LIMIAR_ESTADO)
        print(f"\nLimiar estadual adaptativo: alerta quando a previsão do dia supera {100 * limiar_est:.1f}% "
              f"das previsões dos {JANELA_LIMIAR_ESTADO_DIAS} dias anteriores (percentil escolhido na validação)")
        texto_limiar = "limiar adaptativo"
    else:
        coluna_estado = "probabilidade_estado"
        limiar_est = escolher_limiar(estado_val["evento_estado"].values,
                                     estado_val["probabilidade_estado"].values, LIMIAR_ESTADO,
                                     CRITERIO_LIMIAR_ESTADO)
        texto_limiar = "limiar fixo"
    imprimir_metricas(f"NÍVEL ESTADUAL - teste (cada linha = um dia em SC; evento = "
                      f"{MINIMO_MUNICIPIOS_EVENTO_ESTADUAL}+ municípios com {nome_evento}; "
                      f"{texto_limiar} por {CRITERIO_LIMIAR_ESTADO})",
                      estado_teste["evento_estado"].values,
                      estado_teste[coluna_estado].values, limiar_est)
    if LIMIAR_ESTADO_ADAPTATIVO:
        _, estado_teste["nivel_risco"], estado_teste["comunicado"] = tabela_niveis_risco(
            estado_teste["evento_estado"].values, estado_teste["percentil_recente"].values, "estado (teste)")
    erro_medio = (estado_teste["municipios_esperados"] - estado_teste["municipios_atingidos"]).abs().mean()
    print(f"Municípios atingidos por dia: erro médio absoluto = {erro_medio:.3f} "
          f"(real médio {estado_teste['municipios_atingidos'].mean():.3f}, "
          f"previsto médio {estado_teste['municipios_esperados'].mean():.3f})")

    # ---- Salvar resultados ----
    prev_teste["alerta"] = (prev_teste["pontuacao"] >= prev_teste["limiar"]).astype(int)
    if ALVO == "chuva_extrema":
        avaliar_associacao_desastres(prev_teste)
    estado_teste["alerta"] = (estado_teste[coluna_estado] >= limiar_est).astype(int)
    prev_teste.to_csv(os.path.join(PASTA_SAIDA, "previsoes_municipios_teste.csv"), index=False)
    por_municipio.to_csv(os.path.join(PASTA_SAIDA, "metricas_por_municipio.csv"), index=False)
    estado_teste.to_csv(os.path.join(PASTA_SAIDA, "previsoes_estado_teste.csv"), index=False)
    salvar_graficos(historico, prev_teste, estado_teste, None if LIMIAR_ESTADO_ADAPTATIVO else limiar_est)
    gerar_graficos_previsoes(prev_teste, estado_teste, PASTA_GRAFICOS, NOMES_NIVEIS_RISCO, COMUNICADO_POR_NIVEL)
    modelo.save(os.path.join(PASTA_SAIDA, "modelo_lstm_sc.keras"))
    print(f"Tabelas e modelo salvos em {PASTA_SAIDA}/")


if __name__ == "__main__":
    caminho_analise, arquivo_analise = iniciar_registro()
    try:
        main()
    except BaseException:
        traceback.print_exc()  # o erro também fica registrado no arquivo
        raise
    finally:
        print(f"\nSaída completa salva em {caminho_analise}")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        arquivo_analise.close()