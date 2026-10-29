import math

import networkx as nx
import pytest

from src.algoritmos import astar, dijkstra, haversine


def grafo_teste():
    """Grafo pequeno com coordenadas reais (perto de Brasília).

    Dois caminhos de A a E:
      A-B-C-E  (mais curto em número de arestas, custo 3 + 3 + 3 = 9)
      A-D-E    (custo 10 + 1 = 11, mais caro)
    O caminho ótimo é A-B-C-E com custo 9.
    """
    G = nx.Graph()
    coords = {
        "A": (-15.780, -47.930),
        "B": (-15.781, -47.900),
        "C": (-15.782, -47.870),
        "D": (-15.800, -47.905),
        "E": (-15.783, -47.840),
    }
    for n, (lat, lon) in coords.items():
        G.add_node(n, y=lat, x=lon)
    # comprimentos definidos explicitamente para tornar o caso de teste determinístico
    G.add_edge("A", "B", length=3.0)
    G.add_edge("B", "C", length=3.0)
    G.add_edge("C", "E", length=3.0)
    G.add_edge("A", "D", length=10.0)
    G.add_edge("D", "E", length=1.0)
    return G


def grafo_linha_com_desvio(n=30):
    """Linha reta de n nós ligados em sequência, com coordenadas reais."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, y=-15.78, x=-47.95 + i * 0.002)
    for i in range(n - 1):
        G.add_edge(i, i + 1, length=haversine(-15.78, -47.95 + i * 0.002, -15.78, -47.95 + (i + 1) * 0.002))
    return G


def test_haversine_zero_para_mesmo_ponto():
    assert haversine(-15.78, -47.93, -15.78, -47.93) == pytest.approx(0)


def test_haversine_um_grau_de_latitude_aprox_111km():
    assert haversine(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)


@pytest.mark.parametrize("algoritmo", [dijkstra, astar])
def test_encontra_caminho_otimo(algoritmo):
    res = algoritmo(grafo_teste(), "A", "E")
    assert res.caminho == ["A", "B", "C", "E"]
    assert res.custo == pytest.approx(9.0)


@pytest.mark.parametrize("algoritmo", [dijkstra, astar])
def test_origem_igual_destino(algoritmo):
    res = algoritmo(grafo_teste(), "C", "C")
    assert res.caminho == ["C"]
    assert res.custo == 0


@pytest.mark.parametrize("algoritmo", [dijkstra, astar])
def test_destino_inalcancavel(algoritmo):
    G = grafo_teste()
    G.add_node("Z", y=-15.5, x=-47.5)  # sem arestas
    res = algoritmo(G, "A", "Z")
    assert res.caminho == []
    assert res.custo == math.inf


def test_dijkstra_e_astar_dao_mesmo_custo_otimo():
    G = grafo_linha_com_desvio()
    assert dijkstra(G, 0, 29).custo == pytest.approx(astar(G, 0, 29).custo)


def test_astar_nao_visita_mais_nos_que_dijkstra_em_linha():
    # Em uma linha reta, a heurística aponta exatamente para o destino,
    # então A* deve extrair o mínimo possível de nós.
    G = grafo_linha_com_desvio()
    d = dijkstra(G, 0, 29)
    a = astar(G, 0, 29)
    assert len(a.visitados) <= len(d.visitados)
    assert len(a.visitados) == len(a.caminho)  # nenhum desvio em linha reta


def test_nos_descartados_sao_visitados_fora_do_caminho():
    res = astar(grafo_teste(), "A", "E")
    assert res.nos_descartados == len(res.visitados) - len(res.caminho)
    assert set(res.caminho) <= res.visitados
    assert res.nos_descartados >= 0


def test_relaxacoes_registram_pares_u_v():
    res = dijkstra(grafo_teste(), "A", "E")
    assert len(res.relaxacoes) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in res.relaxacoes)


def test_grafo_direcionado_respeita_sentido_das_arestas():
    G = nx.DiGraph()
    G.add_node("A", y=0, x=0)
    G.add_node("B", y=0, x=1)
    G.add_edge("A", "B", length=1.0)
    assert dijkstra(G, "A", "B").custo == 1.0
    assert dijkstra(G, "B", "A").custo == math.inf


def test_multigrafo_usa_aresta_paralela_mais_barata():
    G = nx.MultiDiGraph()
    G.add_node("A", y=0, x=0)
    G.add_node("B", y=0, x=1)
    G.add_edge("A", "B", length=5.0)
    G.add_edge("A", "B", length=2.0)
    assert dijkstra(G, "A", "B").custo == 2.0
    assert astar(G, "A", "B").custo == 2.0
