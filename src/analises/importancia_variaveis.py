"""
Qual variável é mais responsável pelos desastres?
=================================================

Dois testes, no período de TREINO, com o mesmo alvo do modelo (desastre em
algum dos próximos HORIZONTE_PREVISAO dias):

  1. Univariado: cada variável sozinha. AUC (quanto ela separa dias com e sem
     desastre) e teste de Mann-Whitney (as distribuições diferem?).
  2. Multivariado: regressão logística com todas as variáveis juntas, mais o
     controle do município (taxa histórica de desastre) e da época do ano.
     Mostra o efeito de cada variável DESCONTANDO as outras:
       - odds ratio por 1 desvio padrão (quanto a chance de desastre muda
         quando a variável aumenta 1 desvio padrão, com as outras fixas);
       - teste da razão de verossimilhança (LR): o modelo piora de forma
         significativa sem a variável? O qui-quadrado do teste mede o quanto
         de informação exclusiva cada variável traz.

Para rodar:
    /usr/bin/python3 src/analises/importancia_variaveis.py   (a partir da raiz do repositório)
"""

import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import chi2, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# O modelo fica em src/modelo: essa pasta entra no caminho de importação
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelo"))
import modelo_lstm_desastres as m  # noqa: E402

# #############################################################################
# ##                        PARÂMETROS AJUSTÁVEIS                            ##
# #############################################################################

# Variáveis testadas. Para o teste multivariado, use uma variável por "ideia"
# (ex.: não chuva_acum_7d e chuva_acum_5d juntas), porque variáveis quase
# iguais dividem o efeito entre si e nenhuma parece importante.
VARIAVEIS = [
    "precipitacao",
    "chuva_acum_3d",
    "chuva_acum_7d",
    "chuva_acum_30d",
    "dias_chuva_7d",
    "dias_chuva_15d",
    "dias_chuva_p90_15d",
    "vento_maximo_ms",
    "radiacao_kwh_m2",
]

# Variáveis de chuva em log(1 + x) no teste multivariado (a chuva é muito
# assimétrica: muitos zeros e poucos valores enormes).
VARIAVEIS_LOG = ["precipitacao", "chuva_acum_3d", "chuva_acum_7d", "chuva_acum_30d"]

# Se True, inclui como controles a taxa histórica de desastre do município
# (no treino) e a época do ano. Sem isso, uma variável de chuva pode parecer
# importante só porque chove mais nos municípios que registram mais desastres.
USAR_CONTROLES = True

# Dias sem desastre usados no teste multivariado (todos os dias com desastre
# são usados). Amostrar só os dias sem desastre não altera os odds ratios,
# só o intercepto, e deixa o cálculo bem mais rápido.
DIAS_SEM_DESASTRE = 300_000

# Amostra para o teste de Mann-Whitney.
TAMANHO_AMOSTRA_MW = 300_000

# #############################################################################


def iniciar_registro():
    os.makedirs(m.PASTA_ANALISES, exist_ok=True)
    agora = datetime.now()
    caminho = os.path.join(m.PASTA_ANALISES, f"importancia_{agora:%Y-%m-%d_%H-%M-%S}.txt")
    arquivo = open(caminho, "w", encoding="utf-8")
    sys.stdout = m.Duplicador(sys.__stdout__, arquivo)
    sys.stderr = m.Duplicador(sys.__stderr__, arquivo)
    print(f"Teste de importância das variáveis iniciado em {agora:%d/%m/%Y %H:%M:%S}")
    print(f"  HORIZONTE_PREVISAO = {m.HORIZONTE_PREVISAO} | DATA_FIM_TREINO = {m.DATA_FIM_TREINO!r} | "
          f"USAR_CONTROLES = {USAR_CONTROLES}")
    print("-" * 70)
    return caminho, arquivo


def preparar_dados():
    df = m.carregar_dados()
    fim_treino, _ = m.definir_datas_corte(df)
    df, _ = m.adicionar_variaveis_chuva(df, fim_treino)
    finais, alvo = m.calcular_janelas_validas(df)

    variaveis = [v for v in VARIAVEIS if v in df.columns]
    tabela = df.iloc[finais][["municipio_id", m.COLUNA_DATA] + variaveis].reset_index(drop=True)
    tabela["desastre"] = alvo[finais].astype(int)
    tabela = tabela[tabela[m.COLUNA_DATA] <= fim_treino].reset_index(drop=True)

    taxa_municipio = tabela.groupby("municipio_id")["desastre"].mean()
    tabela["taxa_hist_municipio"] = np.log(tabela["municipio_id"].map(taxa_municipio) + 1e-4)
    dia_ano = tabela[m.COLUNA_DATA].dt.dayofyear
    tabela["sazon_sen"] = np.sin(2 * np.pi * dia_ano / 365.25)
    tabela["sazon_cos"] = np.cos(2 * np.pi * dia_ano / 365.25)

    print(f"Período de treino: {len(tabela):,} município-dias, {tabela['desastre'].sum():,} com desastre "
          f"({100 * tabela['desastre'].mean():.2f}%)")
    return tabela, variaveis


def teste_univariado(tabela, variaveis):
    print(f"\n{'=' * 70}\n1) TESTE UNIVARIADO (cada variável sozinha)\n{'=' * 70}")
    amostra = tabela.sample(min(TAMANHO_AMOSTRA_MW, len(tabela)), random_state=m.SEMENTE)
    linhas = []
    for v in variaveis:
        y, x = tabela["desastre"].values, tabela[v].values
        auc = roc_auc_score(y, x)
        xa, ya = amostra[v].values, amostra["desastre"].values
        _, p = mannwhitneyu(xa[ya == 1], xa[ya == 0], alternative="two-sided")
        linhas.append({"variavel": v, "auc": auc, "separacao": abs(auc - 0.5),
                       "direcao": "maior -> mais desastre" if auc > 0.5 else "menor -> mais desastre",
                       "mediana_desastre": np.median(x[y == 1]), "mediana_normal": np.median(x[y == 0]),
                       "p_mann_whitney": p})
    resultado = pd.DataFrame(linhas).sort_values("separacao", ascending=False)
    print(resultado.drop(columns="separacao").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nAUC: 0,5 = não separa; quanto mais longe de 0,5, mais a variável sozinha separa os dias.")
    return resultado


def ajustar(X, y):
    modelo = LogisticRegression(penalty=None, max_iter=2000)
    modelo.fit(X, y)
    p = np.clip(modelo.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    log_verossimilhanca = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return modelo, log_verossimilhanca


def teste_multivariado(tabela, variaveis):
    print(f"\n{'=' * 70}\n2) TESTE MULTIVARIADO (regressão logística, todas as variáveis juntas)\n{'=' * 70}")
    positivos = tabela[tabela["desastre"] == 1]
    negativos = tabela[tabela["desastre"] == 0]
    negativos = negativos.sample(min(DIAS_SEM_DESASTRE, len(negativos)), random_state=m.SEMENTE)
    dados = pd.concat([positivos, negativos])

    X = dados[variaveis].astype(float).copy()
    for v in VARIAVEIS_LOG:
        if v in X:
            X[v] = np.log1p(X[v].clip(lower=0))
    controles = ["taxa_hist_municipio", "sazon_sen", "sazon_cos"] if USAR_CONTROLES else []
    X[controles] = dados[controles]
    X = (X - X.mean()) / X.std()  # padronizado: coeficientes comparáveis entre variáveis
    y = dados["desastre"].values
    print(f"Amostra: {y.sum():,} dias com desastre e {len(y) - y.sum():,} sem desastre"
          + (f" | controles: {controles}" if controles else ""))

    completo, ll_completo = ajustar(X.values, y)
    coeficientes = dict(zip(X.columns, completo.coef_[0]))

    linhas = []
    for v in variaveis:
        _, ll_sem = ajustar(X.drop(columns=v).values, y)
        estatistica = 2 * (ll_completo - ll_sem)
        linhas.append({"variavel": v,
                       "odds_ratio_por_1dp": np.exp(coeficientes[v]),
                       "lr_qui_quadrado": estatistica,
                       "p_valor": chi2.sf(estatistica, df=1)})
    resultado = pd.DataFrame(linhas).sort_values("lr_qui_quadrado", ascending=False)
    resultado["participacao_pct"] = 100 * resultado["lr_qui_quadrado"] / resultado["lr_qui_quadrado"].sum()
    print(resultado.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nodds_ratio_por_1dp: >1 = aumentar a variável aumenta a chance de desastre, com as outras fixas "
          "(ex.: 1,30 = +30% de chance por desvio padrão); <1 = diminui.")
    print("lr_qui_quadrado: quanto o modelo piora sem a variável (informação exclusiva dela). "
          "participacao_pct: parte do total de informação exclusiva que cabe a cada variável.")

    if controles:
        print("\nControles (efeito por 1 desvio padrão): " + ", ".join(
            f"{c} OR = {np.exp(coeficientes[c]):.2f}" for c in controles))
        _, ll_so_controles = ajustar(X[controles].values, y)
        ganho = 2 * (ll_completo - ll_so_controles)
        print(f"Todas as variáveis meteorológicas juntas, além dos controles: LR qui-quadrado = {ganho:,.0f}, "
              f"gl = {len(variaveis)}, p = {chi2.sf(ganho, df=len(variaveis)):.3g}")

    resultado.to_csv(os.path.join(m.PASTA_SAIDA, "importancia_variaveis.csv"), index=False)
    print(f"\nTabela salva em {m.PASTA_SAIDA}/importancia_variaveis.csv")
    return resultado


def main():
    os.makedirs(m.PASTA_SAIDA, exist_ok=True)
    tabela, variaveis = preparar_dados()
    teste_univariado(tabela, variaveis)
    teste_multivariado(tabela, variaveis)


if __name__ == "__main__":
    caminho_registro, arquivo_registro = iniciar_registro()
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        print(f"\nSaída completa salva em {caminho_registro}")
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        arquivo_registro.close()
