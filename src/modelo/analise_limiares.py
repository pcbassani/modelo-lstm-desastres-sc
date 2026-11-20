"""
Análise de limiares e combinações de variáveis que antecedem desastres
======================================================================

Análise exploratória em três etapas, feita antes de (e para orientar) uma
triagem em cascata no modelo LSTM:

  1. Uma variável por vez: para cada variável, a taxa de desastre em faixas de
     valor e o VALOR MÍNIMO a partir do qual o risco fica FATOR_RISCO_LIMIAR
     vezes acima da média. Os dias acima dos limiares das variáveis mais
     fortes são marcados como "suspeitos" (a triagem).
  2. Combinações, só nos dias suspeitos: uma árvore de decisão rasa gera
     regras legíveis ("A acima de x E B acima de y -> z% de desastre") e a
     correlação entre as variáveis é comparada entre dias com e sem desastre.
  3. Dias normais (os que não passaram da triagem): quantos desastres
     escapam da triagem e como as variáveis se distribuem em cada grupo.

Os limiares e as regras são calculados só no período de TREINO e conferidos
no período de TESTE, com as mesmas datas de corte do modelo.

Usa a leitura de dados, o cache e os parâmetros de modelo_lstm_desastres.py
(arquivo, PERCENTUAL_DADOS, HORIZONTE_PREVISAO, datas de corte etc.). O evento
analisado é o mesmo alvo do modelo: desastre em algum dos próximos
HORIZONTE_PREVISAO dias.

Para rodar:
    /usr/bin/python3 analise_limiares.py
"""

import os
import sys
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

import modelo_lstm_desastres as m

# #############################################################################
# ##                        PARÂMETROS AJUSTÁVEIS                            ##
# #############################################################################

# Variáveis analisadas. As que não existirem nos dados são ignoradas. As de
# chuva acumulada longa e de percentil no município são criadas a partir dos
# parâmetros CHUVA_ACUMULADA_LONGA e COLUNAS_PERCENTIL_MUNICIPIO do modelo.
VARIAVEIS = m.COLUNAS_ENTRADA + [
    "chuva_acum_15d",
    "chuva_acum_30d",
    "precipitacao_pct_mun",
    "chuva_acum_3d_pct_mun",
    "chuva_acum_7d_pct_mun",
    "chuva_acum_30d_pct_mun",
]

# Número de faixas (quantis) usadas para mostrar a taxa de desastre por valor.
# 10 = decis. Variáveis com muitos zeros (como a chuva diária) podem ficar com
# menos faixas, porque faixas repetidas são juntadas.
NUMERO_FAIXAS = 10

# O limiar de uma variável é o MENOR valor a partir do qual a taxa de desastre
# fica pelo menos este número de vezes acima da taxa média. Valores maiores
# geram limiares mais altos: triagem mais seletiva, mas que deixa escapar mais
# desastres. Sugestão de teste: 1.5, 2, 3.
FATOR_RISCO_LIMIAR = 2.0

# Quantis testados como candidatos a limiar (do mediano ao quase máximo).
QUANTIS_CANDIDATOS = np.round(np.arange(0.50, 0.995, 0.01), 3)

# Quantas variáveis (as de maior poder de separação, AUC univariado) formam a
# triagem. Um dia é "suspeito" quando passa do limiar de pelo menos uma delas.
# Mais variáveis = triagem com mais cobertura e menos seletiva.
VARIAVEIS_NA_TRIAGEM = 3

# Se True, só uma variável de cada família de chuva entra na triagem (ex.: não
# usa chuva_acum_7d e chuva_acum_7d_pct_mun ao mesmo tempo), para a triagem
# combinar informações diferentes em vez de repetir a mesma.
TRIAGEM_SEM_REPETIR_FAMILIA = True

# Árvore de decisão da etapa 2: profundidade máxima (número de condições por
# regra) e número mínimo de dias em cada folha. Profundidade 2-3 dá regras
# fáceis de ler; folhas grandes evitam regras baseadas em poucos casos.
PROFUNDIDADE_ARVORE = 3
MINIMO_DIAS_FOLHA = 2000

# Amostra usada nos cálculos mais pesados (correlação e teste de Mann-Whitney),
# para a análise rodar em poucos minutos. None = usa todos os dias.
TAMANHO_AMOSTRA = 300_000

# Pasta dos resultados (tabelas e gráficos).
PASTA_RESULTADOS = os.path.join(m.PASTA_SAIDA, "analise_limiares")

# #############################################################################
# ##                         FIM DOS PARÂMETROS                              ##
# #############################################################################

sorteador = np.random.RandomState(m.SEMENTE)


def iniciar_registro():
    """Copia tudo o que aparece no terminal para analises_anteriores/."""
    os.makedirs(m.PASTA_ANALISES, exist_ok=True)
    agora = datetime.now()
    caminho = os.path.join(m.PASTA_ANALISES, f"limiares_{agora:%Y-%m-%d_%H-%M-%S}.txt")
    arquivo = open(caminho, "w", encoding="utf-8")
    sys.stdout = m.Duplicador(sys.__stdout__, arquivo)
    sys.stderr = m.Duplicador(sys.__stderr__, arquivo)
    print(f"Análise de limiares iniciada em {agora:%d/%m/%Y %H:%M:%S}")
    for nome in ["CAMINHO_DADOS", "PERCENTUAL_DADOS", "HORIZONTE_PREVISAO", "DATA_FIM_TREINO",
                 "DATA_FIM_VALIDACAO", "LIMIAR_DESASTRES"]:
        print(f"  {nome} = {getattr(m, nome)!r}")
    for nome in ["NUMERO_FAIXAS", "FATOR_RISCO_LIMIAR", "VARIAVEIS_NA_TRIAGEM",
                 "TRIAGEM_SEM_REPETIR_FAMILIA", "PROFUNDIDADE_ARVORE", "MINIMO_DIAS_FOLHA"]:
        print(f"  {nome} = {globals()[nome]!r}")
    print("-" * 70)
    return caminho, arquivo


def titulo(texto):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def amostrar(tabela):
    if TAMANHO_AMOSTRA and len(tabela) > TAMANHO_AMOSTRA:
        return tabela.sample(TAMANHO_AMOSTRA, random_state=m.SEMENTE)
    return tabela


def familia(variavel):
    """Agrupa variáveis que medem a mesma coisa (ex.: chuva_acum_7d e seu percentil)."""
    return variavel.replace("_pct_mun", "")


# -----------------------------------------------------------------------------
# Preparação: mesmos dados, alvo e datas de corte do modelo
# -----------------------------------------------------------------------------

def preparar_dados():
    df = m.carregar_dados()
    fim_treino, fim_val = m.definir_datas_corte(df)
    print(f"Treino até {fim_treino:%d/%m/%Y} | validação até {fim_val:%d/%m/%Y} | teste depois")
    df, _ = m.adicionar_variaveis_chuva(df, fim_treino)
    finais, alvo = m.calcular_janelas_validas(df)

    variaveis = [v for v in dict.fromkeys(VARIAVEIS) if v in df.columns]
    ignoradas = [v for v in dict.fromkeys(VARIAVEIS) if v not in df.columns]
    if ignoradas:
        print(f"Variáveis ignoradas (não existem nos dados): {ignoradas}")

    tabela = df.iloc[finais][[m.COLUNA_MUNICIPIO, m.COLUNA_DATA] + variaveis].reset_index(drop=True)
    tabela["desastre"] = alvo[finais].astype(int)
    tabela["periodo"] = np.select(
        [tabela[m.COLUNA_DATA] <= fim_treino, tabela[m.COLUNA_DATA] <= fim_val],
        ["treino", "validação"], "teste")

    print(f"\nEvento analisado: desastre em algum dos próximos {m.HORIZONTE_PREVISAO} dias")
    for periodo in ["treino", "validação", "teste"]:
        parte = tabela[tabela["periodo"] == periodo]
        print(f"  {periodo:>10}: {len(parte):>9,} município-dias, {parte['desastre'].sum():>6,} "
              f"com desastre ({100 * parte['desastre'].mean():.2f}%)")
    return tabela, variaveis


# -----------------------------------------------------------------------------
# Etapa 1: uma variável por vez
# -----------------------------------------------------------------------------

def taxa_por_faixa(x, y, variavel):
    """Taxa de desastre em cada faixa de quantis de uma variável."""
    try:
        faixas = pd.qcut(x, NUMERO_FAIXAS, duplicates="drop")
    except ValueError:  # variável praticamente constante: uma faixa só
        faixas = pd.Series(["todos os valores"] * len(x))
    tabela = pd.DataFrame({"faixa": faixas, "desastre": y}).groupby("faixa", observed=True)["desastre"]
    resultado = tabela.agg(dias="size", desastres="sum").reset_index()
    resultado["taxa_pct"] = 100 * resultado["desastres"] / resultado["dias"]
    resultado["risco_vs_media"] = resultado["taxa_pct"] / (100 * y.mean())
    resultado.insert(0, "variavel", variavel)
    resultado["faixa"] = resultado["faixa"].astype(str)
    return resultado


def encontrar_limiar(x, y):
    """Menor valor a partir do qual a taxa de desastre é FATOR_RISCO_LIMIAR x a média."""
    taxa_media = y.mean()
    for valor in np.unique(np.quantile(x, QUANTIS_CANDIDATOS)):
        acima = x >= valor
        if acima.all() or not acima.any():
            continue
        if y[acima].mean() >= FATOR_RISCO_LIMIAR * taxa_media:
            return valor
    return np.nan


def resumo_limiar(x, y, limiar):
    """Cobertura, custo e risco de um limiar num conjunto de dias."""
    acima = x >= limiar
    taxa_acima = y[acima].mean() if acima.any() else np.nan
    taxa_abaixo = y[~acima].mean() if (~acima).any() else np.nan
    return {
        "dias_acima_pct": 100 * acima.mean(),
        "desastres_acima_pct": 100 * acima[y == 1].mean(),
        "normais_acima_pct": 100 * acima[y == 0].mean(),
        "taxa_acima_pct": 100 * taxa_acima,
        "taxa_abaixo_pct": 100 * taxa_abaixo,
        "risco_acima_vs_abaixo": taxa_acima / taxa_abaixo if taxa_abaixo else np.nan,
    }


def etapa_1(tabela, variaveis):
    titulo("ETAPA 1 - UMA VARIÁVEL POR VEZ (limiares calculados no treino)")
    treino = tabela[tabela["periodo"] == "treino"]
    teste = tabela[tabela["periodo"] == "teste"]
    amostra = amostrar(treino)

    linhas, faixas = [], []
    for v in variaveis:
        x, y = treino[v].values, treino["desastre"].values
        auc = roc_auc_score(y, x)
        xa, ya = amostra[v].values, amostra["desastre"].values
        _, p = mannwhitneyu(xa[ya == 1], xa[ya == 0], alternative="two-sided")
        limiar = encontrar_limiar(x, y)
        linha = {
            "variavel": v,
            "auc_univariado": auc,
            "mediana_desastre": np.median(x[y == 1]),
            "mediana_normal": np.median(x[y == 0]),
            "p_mann_whitney": p,
            "limiar": limiar,
        }
        if np.isfinite(limiar):
            for chave, valor in resumo_limiar(x, y, limiar).items():
                linha[f"treino_{chave}"] = valor
            for chave, valor in resumo_limiar(teste[v].values, teste["desastre"].values, limiar).items():
                linha[f"teste_{chave}"] = valor
        linhas.append(linha)
        faixas.append(taxa_por_faixa(x, y, v))

    resultado = pd.DataFrame(linhas).sort_values("auc_univariado", ascending=False)
    faixas = pd.concat(faixas, ignore_index=True)

    print("AUC univariado: chance de um dia com desastre ter valor maior que um dia normal "
          "(0,5 = a variável não separa; acima de 0,5 = valores altos antecedem desastres).")
    print(f"Limiar: menor valor a partir do qual a taxa de desastre é {FATOR_RISCO_LIMIAR}x a média.\n")
    colunas = ["variavel", "auc_univariado", "mediana_desastre", "mediana_normal", "p_mann_whitney",
               "limiar", "treino_desastres_acima_pct", "treino_normais_acima_pct",
               "treino_risco_acima_vs_abaixo", "teste_desastres_acima_pct", "teste_risco_acima_vs_abaixo"]
    print(resultado[[c for c in colunas if c in resultado.columns]].to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nColunas: *_desastres_acima_pct = % dos desastres acima do limiar (cobertura); "
          "*_normais_acima_pct = % dos dias normais acima (custo); risco_acima_vs_abaixo = "
          "quantas vezes a taxa de desastre acima do limiar supera a de abaixo.")

    resultado.to_csv(os.path.join(PASTA_RESULTADOS, "etapa1_limiares.csv"), index=False)
    faixas.to_csv(os.path.join(PASTA_RESULTADOS, "etapa1_taxa_por_faixa.csv"), index=False)
    grafico_faixas(faixas, resultado)
    return resultado


def grafico_faixas(faixas, resultado):
    principais = resultado["variavel"].head(6).tolist()
    fig, eixos = plt.subplots(2, 3, figsize=(16, 8))
    for ax, v in zip(eixos.ravel(), principais):
        parte = faixas[faixas["variavel"] == v]
        ax.bar(range(len(parte)), parte["taxa_pct"], color="#4C78A8")
        ax.axhline(parte["desastres"].sum() / parte["dias"].sum() * 100, color="gray",
                   linestyle="--", label="média")
        ax.set_xticks(range(len(parte)))
        ax.set_xticklabels(parte["faixa"], rotation=60, ha="right", fontsize=7)
        limiar = resultado.loc[resultado["variavel"] == v, "limiar"].iloc[0]
        ax.set_title(f"{v}\nlimiar = {limiar:.3g}" if np.isfinite(limiar) else v, fontsize=10)
        ax.set_ylabel("% de dias com desastre")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_RESULTADOS, "etapa1_taxa_por_faixa.png"), dpi=120)
    plt.close(fig)


def montar_triagem(tabela, resultado):
    titulo("TRIAGEM - dias 'suspeitos' (acima do limiar de pelo menos uma variável)")
    candidatas = resultado[np.isfinite(resultado["limiar"]) & (resultado["auc_univariado"] > 0.5)]
    escolhidas, familias = [], set()
    for _, linha in candidatas.iterrows():
        if TRIAGEM_SEM_REPETIR_FAMILIA and familia(linha["variavel"]) in familias:
            continue
        escolhidas.append((linha["variavel"], linha["limiar"]))
        familias.add(familia(linha["variavel"]))
        if len(escolhidas) == VARIAVEIS_NA_TRIAGEM:
            break
    if not escolhidas:
        raise RuntimeError("Nenhuma variável com limiar encontrado. Diminua FATOR_RISCO_LIMIAR.")

    print("Regra da triagem: dia suspeito se " + " OU ".join(f"{v} >= {l:.3g}" for v, l in escolhidas))
    suspeito = np.zeros(len(tabela), dtype=bool)
    for v, l in escolhidas:
        suspeito |= tabela[v].values >= l
    tabela["suspeito"] = suspeito

    linhas = []
    for periodo in ["treino", "validação", "teste"]:
        parte = tabela[tabela["periodo"] == periodo]
        s, y = parte["suspeito"].values, parte["desastre"].values
        linhas.append({
            "periodo": periodo,
            "dias_suspeitos_pct": 100 * s.mean(),
            "desastres_capturados_pct": 100 * s[y == 1].mean(),
            "taxa_nos_suspeitos_pct": 100 * y[s].mean(),
            "taxa_nos_normais_pct": 100 * y[~s].mean(),
            "risco_suspeito_vs_normal": y[s].mean() / y[~s].mean(),
        })
    resumo = pd.DataFrame(linhas)
    print(resumo.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\ndesastres_capturados_pct = cobertura da triagem: desastres que chegariam à segunda etapa. "
          "Os que ficam de fora seriam perdidos numa cascata.")
    resumo.to_csv(os.path.join(PASTA_RESULTADOS, "triagem_resumo.csv"), index=False)
    return [v for v, _ in escolhidas]


# -----------------------------------------------------------------------------
# Etapa 2: combinações de variáveis nos dias suspeitos
# -----------------------------------------------------------------------------

def etapa_2(tabela, variaveis):
    titulo("ETAPA 2 - COMBINAÇÕES DE VARIÁVEIS NOS DIAS SUSPEITOS")
    suspeitos = tabela[tabela["suspeito"]]
    treino = suspeitos[suspeitos["periodo"] == "treino"]
    teste = suspeitos[suspeitos["periodo"] == "teste"]

    arvore = DecisionTreeClassifier(max_depth=PROFUNDIDADE_ARVORE, min_samples_leaf=MINIMO_DIAS_FOLHA,
                                    class_weight="balanced", random_state=m.SEMENTE)
    arvore.fit(treino[variaveis].values, treino["desastre"].values)

    print("Regras da árvore de decisão (treinada nos dias suspeitos do treino):\n")
    print(export_text(arvore, feature_names=variaveis, decimals=2, show_weights=False))

    # Taxa real de desastre em cada folha (regra), no treino e no teste
    folhas = []
    for nome, parte in [("treino", treino), ("teste", teste)]:
        folha = arvore.apply(parte[variaveis].values)
        agrupado = pd.DataFrame({"folha": folha, "desastre": parte["desastre"].values}) \
            .groupby("folha")["desastre"].agg(["size", "mean"])
        agrupado.columns = [f"dias_{nome}", f"taxa_{nome}_pct"]
        agrupado[f"taxa_{nome}_pct"] *= 100
        folhas.append(agrupado)
    folhas = pd.concat(folhas, axis=1)
    folhas.insert(0, "regra", [descrever_regra(arvore, variaveis, f) for f in folhas.index])
    folhas = folhas.sort_values("taxa_treino_pct", ascending=False)
    print("Taxa de desastre por regra (folha), do maior para o menor risco:")
    print(folhas.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    folhas.to_csv(os.path.join(PASTA_RESULTADOS, "etapa2_regras.csv"), index=False)

    if teste["desastre"].nunique() == 2:
        auc = roc_auc_score(teste["desastre"], arvore.predict_proba(teste[variaveis].values)[:, 1])
        print(f"\nAUC da árvore nos dias suspeitos do teste: {auc:.3f}")

    importancia = pd.Series(arvore.feature_importances_, index=variaveis).sort_values(ascending=False)
    print("\nImportância das variáveis na árvore:")
    print(importancia[importancia > 0].to_string(float_format=lambda v: f"{v:.3f}"))

    comparar_correlacoes(treino, variaveis)


def descrever_regra(arvore, variaveis, folha):
    """Texto com as condições do caminho da raiz até uma folha."""
    estrutura = arvore.tree_
    pais = {}
    for no in range(estrutura.node_count):
        for filho, lado in [(estrutura.children_left[no], "<="), (estrutura.children_right[no], ">")]:
            if filho != -1:
                pais[filho] = (no, lado)
    condicoes, no = [], folha
    while no in pais:
        pai, lado = pais[no]
        condicoes.append(f"{variaveis[estrutura.feature[pai]]} {lado} {estrutura.threshold[pai]:.3g}")
        no = pai
    return " E ".join(reversed(condicoes)) or "todos os dias"


def comparar_correlacoes(treino, variaveis):
    amostra = amostrar(treino)
    corr_desastre = amostra.loc[amostra["desastre"] == 1, variaveis].corr(method="spearman")
    corr_normal = amostra.loc[amostra["desastre"] == 0, variaveis].corr(method="spearman")
    diferenca = corr_desastre - corr_normal

    pares = []
    for i, a in enumerate(variaveis):
        for b in variaveis[i + 1:]:
            if familia(a) != familia(b):
                pares.append((a, b, corr_desastre.loc[a, b], corr_normal.loc[a, b], diferenca.loc[a, b]))
    pares = pd.DataFrame(pares, columns=["variavel_a", "variavel_b", "corr_dias_desastre",
                                         "corr_dias_normais", "diferenca"])
    pares = pares.reindex(pares["diferenca"].abs().sort_values(ascending=False).index)
    print("\nPares de variáveis cuja correlação (Spearman) mais muda entre dias com desastre e "
          "dias normais, entre os suspeitos do treino:")
    print(pares.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    pares.to_csv(os.path.join(PASTA_RESULTADOS, "etapa2_correlacoes.csv"), index=False)

    fig, eixos = plt.subplots(1, 2, figsize=(18, 8))
    for ax, matriz, nome in [(eixos[0], corr_desastre, "dias com desastre"),
                             (eixos[1], corr_normal, "dias normais")]:
        imagem = ax.imshow(matriz.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(variaveis)))
        ax.set_xticklabels(variaveis, rotation=90, fontsize=7)
        ax.set_yticks(range(len(variaveis)))
        ax.set_yticklabels(variaveis, fontsize=7)
        ax.set_title(f"Correlação de Spearman - {nome} (suspeitos, treino)")
    fig.colorbar(imagem, ax=eixos, shrink=0.7)
    plt.savefig(os.path.join(PASTA_RESULTADOS, "etapa2_correlacoes.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Etapa 3: dias normais e desastres que escapam da triagem
# -----------------------------------------------------------------------------

def etapa_3(tabela, variaveis):
    titulo("ETAPA 3 - DIAS NORMAIS E DESASTRES QUE ESCAPAM DA TRIAGEM")
    treino = tabela[tabela["periodo"] == "treino"]
    grupos = {
        "desastre na triagem": treino[treino["suspeito"] & (treino["desastre"] == 1)],
        "desastre fora da triagem": treino[~treino["suspeito"] & (treino["desastre"] == 1)],
        "dia normal": treino[treino["desastre"] == 0],
    }
    total = (treino["desastre"] == 1).sum()
    escapados = len(grupos["desastre fora da triagem"])
    print(f"Desastres do treino que escapam da triagem: {escapados:,} de {total:,} "
          f"({100 * escapados / total:.1f}%)\n")

    linhas = []
    for v in variaveis:
        linha = {"variavel": v}
        for nome, parte in grupos.items():
            q1, mediana, q3 = np.percentile(parte[v].values, [25, 50, 75]) if len(parte) else (np.nan,) * 3
            linha[f"{nome} (mediana)"] = mediana
            linha[f"{nome} (Q1-Q3)"] = f"{q1:.3g} a {q3:.3g}"
        linhas.append(linha)
    distribuicao = pd.DataFrame(linhas)
    colunas = ["variavel"] + [f"{n} (mediana)" for n in grupos]
    print("Mediana de cada variável por grupo (treino):")
    print(distribuicao[colunas].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    distribuicao.to_csv(os.path.join(PASTA_RESULTADOS, "etapa3_distribuicoes.csv"), index=False)

    fora = grupos["desastre fora da triagem"]
    if len(fora):
        por_municipio = fora.groupby(m.COLUNA_MUNICIPIO, observed=True).size() \
            .sort_values(ascending=False).head(10)
        total_mun = treino[treino["desastre"] == 1].groupby(m.COLUNA_MUNICIPIO, observed=True).size()
        print("\nMunicípios com mais desastres fora da triagem (treino):")
        for municipio, n in por_municipio.items():
            print(f"  {municipio}: {n} de {total_mun.get(municipio, n)} desastres "
                  f"({100 * n / total_mun.get(municipio, n):.0f}%)")
        print("Desastres fora da triagem acontecem sem chuva acima dos limiares: podem ser estiagem, "
              "registros administrativos ou eventos não ligados à chuva do período.")


def main():
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    tabela, variaveis = preparar_dados()
    resultado = etapa_1(tabela, variaveis)
    montar_triagem(tabela, resultado)
    etapa_2(tabela, variaveis)
    etapa_3(tabela, variaveis)
    print(f"\nTabelas e gráficos salvos em {PASTA_RESULTADOS}/")


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
