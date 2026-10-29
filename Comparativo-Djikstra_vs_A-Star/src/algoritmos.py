"""Dijkstra e A* instrumentados para comparação de buscas em grafos de ruas.

Ambas as funções devolvem um ResultadoBusca com o custo ótimo, o caminho,
os nós extraídos (visitados) e todas as relaxações realizadas, permitindo
comparar os dois algoritmos com a mesma métrica.
"""

import heapq
from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt

RAIO_TERRA_M = 6_371_000


@dataclass
class ResultadoBusca:
    custo: float
    caminho: list
    visitados: set = field(default_factory=set)
    relaxacoes: list = field(default_factory=list)

    @property
    def nos_descartados(self):
        """Nós extraídos da fila que não fazem parte do caminho ótimo."""
        return len(self.visitados) - len(self.caminho)


def haversine(lat1, lon1, lat2, lon2):
    """Distância em metros entre dois pontos (grandes círculos)."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * RAIO_TERRA_M * asin(sqrt(a))


def _peso(G, u, v, weight):
    """Peso da aresta u→v. Em multigrafos usa a aresta paralela de menor peso."""
    dados = G[u][v]
    if G.is_multigraph():
        return min(d.get(weight, 1) for d in dados.values())
    return dados.get(weight, 1)


def _vizinhos(G, u):
    return G.successors(u) if G.is_directed() else G.neighbors(u)


def _reconstruir(parent, origem, destino):
    caminho = []
    no = destino
    while no is not None:
        caminho.append(no)
        no = parent[no]
    caminho.reverse()
    return caminho if caminho[0] == origem else []


def dijkstra(G, origem, destino, weight="length"):
    """Caminho mínimo de `origem` a `destino` com Dijkstra instrumentado."""
    visitados = set()
    dist = {n: float("inf") for n in G.nodes()}
    parent = {n: None for n in G.nodes()}
    relaxacoes = []

    dist[origem] = 0
    fila = [(0, origem)]

    while fila:
        d, u = heapq.heappop(fila)
        if u in visitados:
            continue
        visitados.add(u)
        if u == destino:
            break

        for v in _vizinhos(G, u):
            w = _peso(G, u, v, weight)
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                heapq.heappush(fila, (dist[v], v))
                relaxacoes.append((u, v))

    if dist[destino] == float("inf"):
        return ResultadoBusca(float("inf"), [], visitados, relaxacoes)
    return ResultadoBusca(dist[destino], _reconstruir(parent, origem, destino), visitados, relaxacoes)


def astar(G, origem, destino, weight="length"):
    """Caminho mínimo com A*, usando a distância de Haversine até o destino como heurística.

    A heurística é admissível quando os pesos das arestas são distâncias reais
    em metros, pois nenhuma via é mais curta que a linha reta entre seus extremos.
    """
    visitados = set()
    dist = {n: float("inf") for n in G.nodes()}
    parent = {n: None for n in G.nodes()}
    relaxacoes = []

    lat_t, lon_t = G.nodes[destino]["y"], G.nodes[destino]["x"]

    def h(n):
        return haversine(G.nodes[n]["y"], G.nodes[n]["x"], lat_t, lon_t)

    dist[origem] = 0
    fila = [(h(origem), origem)]

    while fila:
        f, u = heapq.heappop(fila)
        if u in visitados:
            continue
        visitados.add(u)
        if u == destino:
            break

        for v in _vizinhos(G, u):
            w = _peso(G, u, v, weight)
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                heapq.heappush(fila, (dist[v] + h(v), v))
                relaxacoes.append((u, v))

    if dist[destino] == float("inf"):
        return ResultadoBusca(float("inf"), [], visitados, relaxacoes)
    return ResultadoBusca(dist[destino], _reconstruir(parent, origem, destino), visitados, relaxacoes)
