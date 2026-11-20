"""
Gera os dados do site (mapa interativo das previsões)
====================================================

Junta a malha municipal do IBGE com as previsões do modelo no período de teste
e grava dois arquivos JavaScript usados pela página docs/index.html:

  docs/site/mapa.js       contornos simplificados dos municípios de SC
  docs/site/previsoes.js  comunicado, percentil e evento de cada município em
                          cada dia, mais o alerta estadual

O site é estático: depois de gerado, basta publicar a pasta docs/ no GitHub
Pages (ou abrir docs/index.html no navegador).

Para rodar, a partir da raiz do repositório (depois de rodar o modelo):
    /usr/bin/python3 src/interface/gerar_site.py

Dependência extra: pyshp (python3 -m pip install pyshp), para ler o shapefile.
"""

import json
import os
import unicodedata

import numpy as np
import pandas as pd
import shapefile

# #############################################################################
# ##                        PARÂMETROS AJUSTÁVEIS                            ##
# #############################################################################

# Shapefile da malha municipal do IBGE (Malha Municipal Digital 2025, SC).
CAMINHO_SHAPEFILE = os.path.expanduser("~/Downloads/SC_Municipios_2025/SC_Municipios_2025.shp")

# Previsões salvas pelo modelo.
PASTA_PREVISOES = "resultados_sc"

# Pasta dos arquivos gerados para o site.
PASTA_SITE = os.path.join("docs", "site")

# Simplificação dos contornos, em graus (0,004° = cerca de 400 m). Valores
# maiores deixam o arquivo menor e o mapa mais "quadrado". O shapefile
# original tem 2,4 milhões de pontos (38 MB); simplificado fica com 1 a 2 MB.
TOLERANCIA_SIMPLIFICACAO = 0.004

# Casas decimais das coordenadas (4 = cerca de 11 m).
CASAS_DECIMAIS = 4

# Níveis de risco, na ordem do modelo, e o comunicado de cada um.
NIVEIS = ["baixo", "médio", "alto", "muito alto"]
COMUNICADO = {"baixo": "observação", "médio": "aviso", "alto": "alerta", "muito alto": "alerta"}

# #############################################################################


def normalizar(nome):
    """Nome sem acentos, minúsculo e sem pontuação, para casar IBGE e planilha."""
    sem_acento = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode()
    return "".join(c for c in sem_acento.lower() if c.isalnum())


def simplificar(pontos, tolerancia):
    """Douglas-Peucker iterativo: mantém só os pontos que dão forma ao contorno."""
    pontos = np.asarray(pontos, dtype=float)
    if len(pontos) <= 4:
        return pontos
    manter = np.zeros(len(pontos), dtype=bool)
    manter[0] = manter[-1] = True
    pilha = [(0, len(pontos) - 1)]
    while pilha:
        inicio, fim = pilha.pop()
        if fim <= inicio + 1:
            continue
        a, b = pontos[inicio], pontos[fim]
        trecho = pontos[inicio + 1:fim]
        dx, dy = b - a
        comprimento = np.hypot(dx, dy)
        if comprimento == 0:
            distancias = np.hypot(*(trecho - a).T)
        else:
            distancias = np.abs(dx * (a[1] - trecho[:, 1]) - dy * (a[0] - trecho[:, 0])) / comprimento
        i = int(np.argmax(distancias))
        if distancias[i] > tolerancia:
            meio = inicio + 1 + i
            manter[meio] = True
            pilha.append((inicio, meio))
            pilha.append((meio, fim))
    resultado = pontos[manter]
    return resultado if len(resultado) >= 4 else pontos[[0, len(pontos) // 3, 2 * len(pontos) // 3, -1]]


def ler_municipios():
    leitor = shapefile.Reader(CAMINHO_SHAPEFILE, encoding="utf-8")
    campos = [f[0] for f in leitor.fields[1:]]
    municipios = []
    for forma, registro in zip(leitor.shapes(), leitor.records()):
        dados = dict(zip(campos, registro))
        partes = list(forma.parts) + [len(forma.points)]
        aneis = []
        for i in range(len(partes) - 1):
            anel = simplificar(forma.points[partes[i]:partes[i + 1]], TOLERANCIA_SIMPLIFICACAO)
            aneis.append(np.round(anel, CASAS_DECIMAIS).tolist())
        municipios.append({"codigo": dados["CD_MUN"], "nome": dados["NM_MUN"],
                           "regiao": dados["NM_RGINT"], "aneis": aneis})
    return municipios


def preparar_previsoes():
    previsoes = pd.read_csv(os.path.join(PASTA_PREVISOES, "previsoes_municipios_teste.csv"),
                            usecols=["municipio", "data", "real", "percentil_recente", "nivel_risco"])
    estado = pd.read_csv(os.path.join(PASTA_PREVISOES, "previsoes_estado_teste.csv"))
    datas = sorted(set(previsoes["data"]) | set(estado["data"]))
    posicao = {d: i for i, d in enumerate(datas)}
    nivel_codigo = {n: str(i) for i, n in enumerate(NIVEIS)}

    por_municipio = {}
    for municipio, g in previsoes.groupby("municipio"):
        nivel = ["-"] * len(datas)
        evento = ["-"] * len(datas)
        percentil = ["--"] * len(datas)
        for d, n, r, p in zip(g["data"], g["nivel_risco"], g["real"], g["percentil_recente"]):
            i = posicao[d]
            nivel[i] = nivel_codigo.get(n, "-")
            evento[i] = str(int(r))
            percentil[i] = f"{min(int(round(100 * p)), 99):02d}"
        por_municipio[normalizar(municipio)] = {"nome": municipio, "nivel": "".join(nivel),
                                                "evento": "".join(evento), "pct": "".join(percentil)}

    estado = estado.set_index("data").reindex(datas)
    dados_estado = {
        "nivel": "".join(nivel_codigo.get(n, "-") if isinstance(n, str) else "-" for n in estado["nivel_risco"]),
        "evento": "".join("-" if pd.isna(e) else str(int(e)) for e in estado["evento_estado"]),
        # O alerta estadual usa o limiar escolhido na validação, não o corte dos níveis
        "alerta": "".join("-" if pd.isna(a) else str(int(a)) for a in estado["alerta"]),
        "atingidos": [None if pd.isna(v) else int(v) for v in estado["municipios_atingidos"]],
        "previstos": [None if pd.isna(v) else round(float(v), 1) for v in estado["municipios_esperados"]],
    }
    return datas, por_municipio, dados_estado, previsoes


def resumo_metricas(previsoes, estado):
    """Desempenho do comunicado de alerta (níveis alto e muito alto) no teste."""
    alerta = previsoes["nivel_risco"].isin(["alto", "muito alto"]).values
    real = previsoes["real"].values == 1
    alerta_estado = np.array([a == "1" for a in estado["alerta"]])
    real_estado = np.array([e == "1" for e in estado["evento"]])
    validos = np.array([e != "-" for e in estado["evento"]])

    def medidas(a, r):
        return {"sensibilidade": round(100 * (a & r).sum() / r.sum(), 1),
                "especificidade": round(100 * (~a & ~r).sum() / (~r).sum(), 1),
                "acuracia": round(100 * (a == r).mean(), 1),
                "precisao": round(100 * (a & r).sum() / max(a.sum(), 1), 1)}

    return {"municipal": medidas(alerta, real),
            "estadual": medidas(alerta_estado[validos], real_estado[validos])}


def main():
    os.makedirs(PASTA_SITE, exist_ok=True)
    print("Lendo e simplificando a malha do IBGE ...")
    municipios = ler_municipios()
    datas, por_municipio, dados_estado, previsoes = preparar_previsoes()

    features, sem_previsao = [], []
    for m in municipios:
        chave = normalizar(m["nome"])
        if chave not in por_municipio:
            sem_previsao.append(m["nome"])
        geometria = {"type": "Polygon", "coordinates": m["aneis"]} if len(m["aneis"]) == 1 else \
            {"type": "MultiPolygon", "coordinates": [[anel] for anel in m["aneis"]]}
        features.append({"type": "Feature", "geometry": geometria,
                         "properties": {"codigo": m["codigo"], "nome": m["nome"], "regiao": m["regiao"],
                                        "chave": chave if chave in por_municipio else None}})
    sem_mapa = sorted(v["nome"] for k, v in por_municipio.items()
                      if k not in {normalizar(m["nome"]) for m in municipios})
    if sem_previsao:
        print(f"Áreas do mapa sem previsão (ficam em cinza claro): {sem_previsao}")
    if sem_mapa:
        print(f"Municípios com previsão que não foram encontrados no mapa: {sem_mapa}")

    with open(os.path.join(PASTA_SITE, "mapa.js"), "w", encoding="utf-8") as arquivo:
        arquivo.write("window.MAPA_SC = ")
        json.dump({"type": "FeatureCollection", "features": features}, arquivo,
                  ensure_ascii=False, separators=(",", ":"))
        arquivo.write(";\n")

    dados = {"datas": datas, "municipios": por_municipio, "estado": dados_estado,
             "metricas": resumo_metricas(previsoes, dados_estado),
             "gerado_em": pd.Timestamp.now().strftime("%d/%m/%Y")}
    with open(os.path.join(PASTA_SITE, "previsoes.js"), "w", encoding="utf-8") as arquivo:
        arquivo.write("window.PREVISOES = ")
        json.dump(dados, arquivo, ensure_ascii=False, separators=(",", ":"))
        arquivo.write(";\n")

    for nome in ["mapa.js", "previsoes.js"]:
        tamanho = os.path.getsize(os.path.join(PASTA_SITE, nome)) / 1e6
        print(f"  {os.path.join(PASTA_SITE, nome)}: {tamanho:.1f} MB")
    print(f"{len(features)} áreas no mapa, {len(por_municipio)} municípios com previsão, {len(datas)} dias")


if __name__ == "__main__":
    main()
