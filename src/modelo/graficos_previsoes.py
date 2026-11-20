"""
Gráficos das previsões do modelo
================================

Gera, em PNG, os gráficos das previsões de eventos do período de teste:
matrizes de confusão, níveis de risco e comunicados, curvas ROC e
precisão-recall, alertas estaduais ao longo do tempo, desempenho por
município e alertas por mês.

O modelo (modelo_lstm_desastres.py) chama gerar_graficos_previsoes() ao final
de cada execução. Para refazer os gráficos a partir das previsões já salvas,
sem treinar o modelo de novo, rode a partir da raiz do repositório:

    /usr/bin/python3 src/modelo/graficos_previsoes.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

# Mesmos valores padrão do modelo; gerar_graficos_previsoes() recebe os do modelo.
PASTA_PREVISOES = "resultados_sc"
PASTA_GRAFICOS = os.path.join("docs", "analises", "graficos")
NOMES_NIVEIS_RISCO = ("baixo", "médio", "alto", "muito alto")
COMUNICADO_POR_NIVEL = {"baixo": "observação", "médio": "aviso", "alto": "alerta", "muito alto": "alerta"}
ORDEM_COMUNICADOS = ("observação", "aviso", "alerta")

CINZA, AZUL, VERMELHO = "#9E9E9E", "#4C78A8", "#E45756"
CORES_COMUNICADO = {"observação": "#9E9E9E", "aviso": "#F2B447", "alerta": "#E45756"}
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def salvar(fig, pasta, nome):
    caminho = os.path.join(pasta, nome)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    print(f"  {caminho}")


def grafico_matrizes(previsoes, estado, pasta):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (titulo, y, pred) in zip(eixos, [
            ("Municipal (cada município em cada dia)", previsoes["real"], previsoes["alerta"]),
            ("Estadual (dias com 10+ municípios atingidos)", estado["evento_estado"], estado["alerta"])]):
        matriz = confusion_matrix(y, pred, labels=[0, 1])
        percentual = matriz / matriz.sum(axis=1, keepdims=True)
        ax.imshow(percentual, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matriz[i, j]:,}".replace(",", ".") + f"\n({100 * percentual[i, j]:.1f}% da linha)",
                        ha="center", va="center", fontsize=10,
                        color="white" if percentual[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["previu sem evento", "previu evento"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["sem evento (real)", "com evento (real)"])
        sens = matriz[1, 1] / matriz[1].sum()
        espec = matriz[0, 0] / matriz[0].sum()
        acur = np.trace(matriz) / matriz.sum()
        ax.set_title(f"{titulo}\nsensibilidade {100 * sens:.1f}% | especificidade {100 * espec:.1f}% | "
                     f"acurácia {100 * acur:.1f}%", fontsize=10)
    fig.suptitle("Matrizes de confusão das previsões (teste, 2021-2024)", fontsize=12)
    salvar(fig, pasta, "01_matrizes_confusao.png")


def grafico_niveis(previsoes, pasta):
    y = previsoes["real"].values
    base = y.mean()
    niveis = [n for n in NOMES_NIVEIS_RISCO if n in set(previsoes["nivel_risco"])]
    taxa = [100 * y[previsoes["nivel_risco"].values == n].mean() for n in niveis]
    eventos = [100 * y[previsoes["nivel_risco"].values == n].sum() / y.sum() for n in niveis]
    cores = [CORES_COMUNICADO[COMUNICADO_POR_NIVEL[n]] for n in niveis]

    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = eixos[0]
    barras = ax.bar(niveis, taxa, color=cores)
    ax.axhline(100 * base, color="black", linestyle="--", linewidth=1, label=f"média ({100 * base:.1f}%)")
    for barra, valor in zip(barras, taxa):
        ax.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.1f}%\n({valor / (100 * base):.1f}x)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("chance real de desastre (%)")
    ax.set_title("Chance real de desastre em cada nível de risco")
    ax.set_ylim(0, max(taxa) * 1.25)
    ax.legend()

    ax = eixos[1]
    barras = ax.bar(niveis, eventos, color=cores)
    for barra, valor in zip(barras, eventos):
        ax.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("% de todos os desastres")
    ax.set_title("Onde caem os desastres")
    ax.set_ylim(0, max(eventos) * 1.2)
    legenda = [plt.Rectangle((0, 0), 1, 1, color=CORES_COMUNICADO[c]) for c in ORDEM_COMUNICADOS]
    ax.legend(legenda, ORDEM_COMUNICADOS, title="comunicado")
    fig.suptitle("Níveis de risco e comunicados - municípios (teste, 2021-2024)", fontsize=12)
    salvar(fig, pasta, "02_niveis_risco_comunicados.png")


def grafico_curvas(previsoes, pasta):
    y, p = previsoes["real"].values, previsoes["probabilidade"].values
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    referencias = [("LSTM", p, AZUL)]
    if "clima_prob" in previsoes and previsoes["clima_prob"].notna().all():
        referencias.append(("climatologia (município x mês)", previsoes["clima_prob"].values, CINZA))

    for nome, valores, cor in referencias:
        fpr, tpr, _ = roc_curve(y, valores)
        eixos[0].plot(fpr, tpr, color=cor, label=f"{nome}: AUC {roc_auc_score(y, valores):.3f}")
        precisao, recall, _ = precision_recall_curve(y, valores)
        eixos[1].plot(recall, precisao, color=cor, label=f"{nome}: área {auc(recall, precisao):.3f}")
    eixos[0].plot([0, 1], [0, 1], color="black", linestyle=":", label="acaso: AUC 0,500")
    sens = previsoes.loc[previsoes["real"] == 1, "alerta"].mean()
    fpr_alerta = previsoes.loc[previsoes["real"] == 0, "alerta"].mean()
    eixos[0].scatter([fpr_alerta], [sens], color=VERMELHO, zorder=3, label="ponto do alerta atual")
    eixos[0].set_xlabel("1 - especificidade (dias normais com alerta)")
    eixos[0].set_ylabel("sensibilidade (desastres detectados)")
    eixos[0].set_title("Curva ROC")
    eixos[0].legend(loc="lower right")
    eixos[1].axhline(y.mean(), color="black", linestyle=":", label=f"acaso: {y.mean():.3f}")
    eixos[1].set_xlabel("recall (desastres detectados)")
    eixos[1].set_ylabel("precisão (alertas que se confirmam)")
    eixos[1].set_title("Curva precisão-recall")
    eixos[1].legend(loc="upper right")
    fig.suptitle("Capacidade de separar dias com e sem desastre - municípios (teste)", fontsize=12)
    salvar(fig, pasta, "03_curvas_roc_precisao_recall.png")


def grafico_estado_tempo(estado, pasta):
    estado = estado.sort_values("data")
    datas = pd.to_datetime(estado["data"])
    fig, eixos = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = eixos[0]
    ax.plot(datas, estado["municipios_atingidos"], color=CINZA, linewidth=0.7, label="municípios atingidos (real)")
    ax.plot(datas, estado["municipios_esperados"], color=AZUL, linewidth=0.9, label="municípios previstos (esperado)")
    ax.axhline(10, color="black", linestyle="--", linewidth=0.8, label="10 municípios (evento estadual)")
    ax.set_ylabel("municípios por dia")
    ax.set_title("Municípios atingidos por dia em SC: real x previsto")
    ax.legend(loc="upper left")

    ax = eixos[1]
    if "percentil_recente" in estado:
        ax.plot(datas, estado["percentil_recente"], color=AZUL, linewidth=0.6,
                label="posição da previsão no último ano (percentil)")
    evento = estado["evento_estado"] == 1
    ax.scatter(datas[evento], np.full(evento.sum(), 1.04), color=VERMELHO, s=5, label="dia com evento estadual")
    alerta = estado["alerta"] == 1
    ax.scatter(datas[alerta], np.full(alerta.sum(), 1.08), color="black", s=5, label="dia com alerta estadual")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("percentil")
    ax.set_title("Alertas estaduais e dias com evento")
    ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Previsão estadual ao longo do teste (2021-2024)", fontsize=12)
    salvar(fig, pasta, "04_previsao_estadual_no_tempo.png")


def grafico_municipios(previsoes, pasta, quantidade=15):
    linhas = []
    for municipio, g in previsoes.groupby("municipio"):
        y, a = g["real"].values, g["alerta"].values
        if y.sum() == 0:
            continue
        linhas.append({"municipio": municipio, "eventos": int(y.sum()),
                       "sensibilidade": 100 * a[y == 1].mean(),
                       "especificidade": 100 * (1 - a[y == 0].mean()),
                       "auc": roc_auc_score(y, g["probabilidade"].values) if len(np.unique(y)) == 2 else np.nan})
    tabela = pd.DataFrame(linhas).sort_values("eventos", ascending=False).head(quantidade).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(tabela) + 1.5))
    posicoes = np.arange(len(tabela))
    ax.barh(posicoes - 0.2, tabela["sensibilidade"], 0.4, color=AZUL, label="sensibilidade (desastres detectados)")
    ax.barh(posicoes + 0.2, tabela["especificidade"], 0.4, color=CINZA, label="especificidade (dias normais)")
    ax.axvline(50, color="black", linestyle="--", linewidth=0.8, label="meta de 50%")
    ax.set_yticks(posicoes)
    ax.set_yticklabels([f"{m} ({e} eventos, AUC {a:.2f})" for m, e, a in
                        zip(tabela["municipio"], tabela["eventos"], tabela["auc"])], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.set_title(f"Desempenho nos {quantidade} municípios com mais eventos (teste)")
    ax.legend(loc="lower right", fontsize=8)
    salvar(fig, pasta, "05_desempenho_por_municipio.png")


def grafico_meses(previsoes, pasta):
    mes = pd.to_datetime(previsoes["data"]).dt.month
    taxa_evento = previsoes.groupby(mes)["real"].mean().reindex(range(1, 13)) * 100
    taxa_alerta = previsoes.groupby(mes)["alerta"].mean().reindex(range(1, 13)) * 100
    deteccao = previsoes[previsoes["real"] == 1].groupby(mes[previsoes["real"] == 1])["alerta"].mean() \
        .reindex(range(1, 13)) * 100
    fig, eixos = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(12)
    ax = eixos[0]
    ax.bar(x, taxa_evento, color=VERMELHO)
    for i, v in enumerate(taxa_evento):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("% de município-dias")
    ax.set_title("Frequência real de desastres por mês")
    ax.set_ylim(0, taxa_evento.max() * 1.2)

    ax = eixos[1]
    ax.plot(x, deteccao, color=AZUL, marker="s", label="desastres detectados pelo alerta")
    ax.plot(x, taxa_alerta, color=CINZA, marker="o", label="município-dias com alerta")
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, label="50%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Detecção e volume de alertas por mês")
    ax.set_xticks(x)
    ax.set_xticklabels(MESES)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Desempenho do alerta ao longo do ano - municípios (teste, 2021-2024)", fontsize=12)
    salvar(fig, pasta, "06_alertas_por_mes.png")


def gerar_graficos_previsoes(previsoes, estado, pasta=PASTA_GRAFICOS, nomes_niveis=None,
                             comunicado_por_nivel=None):
    """Gera os gráficos das previsões do teste. previsoes/estado = tabelas salvas pelo modelo."""
    global NOMES_NIVEIS_RISCO, COMUNICADO_POR_NIVEL
    if nomes_niveis:
        NOMES_NIVEIS_RISCO = tuple(nomes_niveis)
    if comunicado_por_nivel:
        COMUNICADO_POR_NIVEL = dict(comunicado_por_nivel)
    os.makedirs(pasta, exist_ok=True)
    print(f"\nGráficos das previsões salvos em {pasta}/:")
    grafico_matrizes(previsoes, estado, pasta)
    if "nivel_risco" in previsoes:
        grafico_niveis(previsoes, pasta)
    grafico_curvas(previsoes, pasta)
    grafico_estado_tempo(estado, pasta)
    grafico_municipios(previsoes, pasta)
    grafico_meses(previsoes, pasta)


if __name__ == "__main__":
    previsoes = pd.read_csv(os.path.join(PASTA_PREVISOES, "previsoes_municipios_teste.csv"))
    estado = pd.read_csv(os.path.join(PASTA_PREVISOES, "previsoes_estado_teste.csv"))
    gerar_graficos_previsoes(previsoes, estado)
