"""
Climatologia: em que meses cada município tem mais chuva extrema e desastres?
============================================================================

Calcula, só com o período de TREINO, a probabilidade de chuva extrema e de
desastre em cada município e em cada mês do ano, e classifica cada mês como
risco BAIXO, MÉDIO ou ALTO em relação à média anual do próprio município.
Depois confere no período de TESTE se os meses de risco alto realmente
tiveram mais eventos (taxa por categoria, qui-quadrado e AUC).

Os dois eventos usam as mesmas definições do modelo (modelo_lstm_desastres.py):
  - desastre: registro de desastre nos próximos HORIZONTE_PREVISAO dias;
  - chuva extrema: chuva dos próximos HORIZONTE_PREVISAO dias acima do limiar
    (TIPO_LIMIAR_CHUVA, PERCENTIL_CHUVA_EXTREMA, LIMIAR_CHUVA_MM).
A suavização e os cortes das categorias também vêm do modelo
(SUAVIZACAO_CLIMATOLOGIA e CORTES_CATEGORIA_CLIMATOLOGIA).

Resultados em resultados_sc/climatologia/:
  - calendario_risco_<evento>.csv : município x mês, com probabilidade, risco
    relativo e categoria (o "calendário de risco");
  - mapa_<evento>.png             : mapa de calor município x mês;
  - perfil_mensal_sc.png          : taxa por mês no estado, treino x teste.

Para rodar:
    /usr/bin/python3 src/analises/climatologia.py   (a partir da raiz do repositório)
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
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

# O modelo fica em src/modelo: essa pasta entra no caminho de importação
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelo"))
import modelo_lstm_desastres as m  # noqa: E402

# #############################################################################
# ##                        PARÂMETROS AJUSTÁVEIS                            ##
# #############################################################################

# Quantos municípios mostrar no terminal com o calendário de risco
# (os que tiveram mais eventos no treino).
MUNICIPIOS_NO_TERMINAL = 10

# Quantos municípios aparecem no mapa de calor (os de mais eventos no treino).
# Use None para todos os 293 (o gráfico fica bem alto).
MUNICIPIOS_NO_MAPA = 60

# Pasta dos resultados.
PASTA_RESULTADOS = os.path.join(m.PASTA_SAIDA, "climatologia")

# #############################################################################

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def iniciar_registro():
    os.makedirs(m.PASTA_ANALISES, exist_ok=True)
    agora = datetime.now()
    caminho = os.path.join(m.PASTA_ANALISES, f"climatologia_{agora:%Y-%m-%d_%H-%M-%S}.txt")
    arquivo = open(caminho, "w", encoding="utf-8")
    sys.stdout = m.Duplicador(sys.__stdout__, arquivo)
    sys.stderr = m.Duplicador(sys.__stderr__, arquivo)
    print(f"Climatologia iniciada em {agora:%d/%m/%Y %H:%M:%S}")
    for nome in ["HORIZONTE_PREVISAO", "TIPO_LIMIAR_CHUVA", "PERCENTIL_CHUVA_EXTREMA", "LIMIAR_CHUVA_MM",
                 "SUAVIZACAO_CLIMATOLOGIA", "CORTES_CATEGORIA_CLIMATOLOGIA", "DATA_FIM_TREINO",
                 "DATA_FIM_VALIDACAO"]:
        print(f"  {nome} = {getattr(m, nome)!r}")
    print("-" * 70)
    return caminho, arquivo



def pasta_graficos():
    """Os gráficos vão para a documentação (docs/analises/graficos); as tabelas, para PASTA_RESULTADOS."""
    os.makedirs(m.PASTA_GRAFICOS, exist_ok=True)
    return m.PASTA_GRAFICOS


def titulo(texto):
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def analisar_evento(df, finais, alvo, fim_treino, fim_val, nome, sufixo):
    titulo(f"CLIMATOLOGIA DE {nome.upper()}")
    clima = m.climatologia_mensal(df, finais, alvo, fim_treino)
    nomes = df.groupby("municipio_id")[m.COLUNA_MUNICIPIO].first()
    clima.insert(0, "municipio", clima["municipio_id"].map(nomes).astype(str))

    # Calendário de risco (salvo para todos os municípios)
    calendario = clima[["municipio", "mes", "dias", "eventos", "probabilidade", "risco_relativo", "categoria"]]
    calendario.to_csv(os.path.join(PASTA_RESULTADOS, f"calendario_risco_{sufixo}.csv"), index=False)

    # Avaliação: a categoria e a probabilidade de cada amostra, por período
    prob, risco, categoria = m.aplicar_climatologia(df, clima)
    datas = df[m.COLUNA_DATA].values[finais]
    periodos = {
        "treino": finais[datas <= np.datetime64(fim_treino)],
        "teste": finais[datas > np.datetime64(fim_val)],
    }
    linhas = []
    for periodo, idx in periodos.items():
        y = alvo[idx]
        for cat in ["baixa", "média", "alta"]:
            marcado = categoria[idx] == cat
            linhas.append({"periodo": periodo, "categoria": cat,
                           "municipio_meses_pct": 100 * marcado.mean(),
                           "taxa_evento_pct": 100 * y[marcado].mean() if marcado.any() else np.nan,
                           "eventos_pct": 100 * y[marcado].sum() / max(y.sum(), 1)})
    tabela = pd.DataFrame(linhas)
    taxa_media = {p: 100 * alvo[idx].mean() for p, idx in periodos.items()}
    tabela["risco_vs_media"] = tabela.apply(lambda r: r["taxa_evento_pct"] / taxa_media[r["periodo"]], axis=1)
    print("Taxa do evento por categoria de risco climatológico (categoria definida no treino):")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("eventos_pct = parte dos eventos do período que caiu em meses daquela categoria.")

    idx = periodos["teste"]
    y = alvo[idx]
    _, p, gl, _ = chi2_contingency(pd.crosstab(categoria[idx], y))
    auc = roc_auc_score(y, prob[idx]) if len(np.unique(y)) == 2 else np.nan
    print(f"\nTeste (2021 em diante): qui-quadrado categoria x evento p = {p:.3g} (gl = {gl}) | "
          f"AUC só com a probabilidade climatológica = {auc:.3f}")
    print("Compare esse AUC com o da LSTM: se for parecido, o modelo aprendeu basicamente a época do ano.")

    # Perfil do estado: taxa por mês no treino e no teste
    mes = df[m.COLUNA_DATA].dt.month.values
    perfil = pd.DataFrame({p: pd.Series(alvo[i]).groupby(mes[i]).mean() * 100 for p, i in periodos.items()})
    perfil = perfil.reindex(range(1, 13))
    perfil.index = MESES
    print("\nTaxa do evento por mês no estado (%):")
    print(perfil.T.to_string(float_format=lambda v: f"{v:.2f}"))

    # Calendário dos municípios com mais eventos
    principais = clima.groupby("municipio")["eventos"].sum().sort_values(ascending=False)
    print(f"\nCalendário de risco dos {MUNICIPIOS_NO_TERMINAL} municípios com mais eventos no treino:")
    for municipio in principais.index[:MUNICIPIOS_NO_TERMINAL]:
        parte = clima[clima["municipio"] == municipio].set_index("mes")
        alto = [MESES[i - 1] for i in parte.index if parte.loc[i, "categoria"] == "alta"]
        baixo = [MESES[i - 1] for i in parte.index if parte.loc[i, "categoria"] == "baixa"]
        print(f"  {municipio}: alto em {', '.join(alto) or '-'} | baixo em {', '.join(baixo) or '-'}")

    salvar_mapa(clima, principais, nome, sufixo)
    return perfil


def salvar_mapa(clima, principais, nome, sufixo):
    escolhidos = principais.index if MUNICIPIOS_NO_MAPA is None else principais.index[:MUNICIPIOS_NO_MAPA]
    matriz = clima[clima["municipio"].isin(escolhidos)] \
        .pivot(index="municipio", columns="mes", values="risco_relativo").reindex(escolhidos)
    altura = max(6, 0.22 * len(matriz))
    fig, ax = plt.subplots(figsize=(10, altura))
    imagem = ax.imshow(matriz.values, cmap="RdYlBu_r", vmin=0, vmax=2.5, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MESES)
    ax.set_yticks(range(len(matriz)))
    ax.set_yticklabels(matriz.index, fontsize=7)
    ax.set_title(f"Risco relativo de {nome} por mês (1 = média do município; treino)")
    fig.colorbar(imagem, ax=ax, shrink=0.6, label="risco relativo")
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos(), f"analises_climatologia_mapa_{sufixo}.png"), dpi=120)
    plt.close(fig)


def salvar_perfil(perfis):
    fig, eixos = plt.subplots(1, len(perfis), figsize=(7 * len(perfis), 4.5))
    for ax, (nome, perfil) in zip(np.atleast_1d(eixos), perfis.items()):
        largura = 0.4
        x = np.arange(12)
        ax.bar(x - largura / 2, perfil["treino"], largura, label="treino (2000-2016)", color="#9E9E9E")
        ax.bar(x + largura / 2, perfil["teste"], largura, label="teste (2021-2024)", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(MESES)
        ax.set_ylabel("% de município-dias com evento")
        ax.set_title(f"{nome.capitalize()} por mês em SC")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_graficos(), "analises_climatologia_perfil_mensal_sc.png"), dpi=120)
    plt.close(fig)


def main():
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    df = m.carregar_dados()
    fim_treino, fim_val = m.definir_datas_corte(df)
    print(f"Treino até {fim_treino:%d/%m/%Y} | validação até {fim_val:%d/%m/%Y} | teste depois")
    finais, alvo_desastre = m.calcular_janelas_validas(df)
    alvo_chuva = m.alvo_chuva_extrema(df, finais, fim_treino)

    perfis = {
        "chuva extrema": analisar_evento(df, finais, alvo_chuva, fim_treino, fim_val, "chuva extrema", "chuva"),
        "desastre": analisar_evento(df, finais, alvo_desastre, fim_treino, fim_val, "desastre", "desastre"),
    }
    salvar_perfil(perfis)
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
