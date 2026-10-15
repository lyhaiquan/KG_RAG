import re, json, itertools, collections
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = set(
    "the and for are was were with that this from what which how does did who "
    "whom into than then have has had its their about between over under after "
    "before while there any".split()
)


def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def chunk_text(title, text, n=3):
    sents = split_sentences(text)
    return [f"{title}. " + " ".join(sents[i : i + n]) for i in range(0, len(sents), n)]


def compile_lexicon(lex):
    return {k: re.compile(v[1], re.I) for k, v in lex.items()}


def extract_entities(text, rx):
    return sorted(k for k, r in rx.items() if r.search(text))


def make_rel(table):
    def rel(t1, t2):
        return table.get((t1, t2)) or table.get((t2, t1)) or "related"

    return rel


def new_kg(lex):
    G = nx.Graph()
    for name, (t, _) in lex.items():
        G.add_node(name, kind="entity", etype=t)
    return G


def add_cooccurrence(G, docs, rel, skip_types=("Year",), doc_key=None, doc_kind="doc"):
    """Quan hệ từ văn bản: 2 thực thể cùng xuất hiện trong 1 tài liệu -> 1 cạnh.

    Trọng số cạnh = số tài liệu mà hai thực thể cùng xuất hiện."""
    for d in docs:
        ents = [e for e in d["entities"] if G.nodes[e]["etype"] not in skip_types]
        if doc_key:
            dn = d[doc_key]
            if dn not in G:
                G.add_node(dn, kind=doc_kind)
            for e in ents:
                G.add_edge(dn, e, rel="MENTIONS", weight=1)
        for a, b in itertools.combinations(ents, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(
                    a,
                    b,
                    rel=rel(G.nodes[a]["etype"], G.nodes[b]["etype"]),
                    weight=1,
                    source="text",
                )
    return G


def graph_to_dict(G):
    return {
        "nodes": [[n, dict(a)] for n, a in G.nodes(data=True)],
        "edges": [[u, v, dict(a)] for u, v, a in G.edges(data=True)],
    }


def graph_from_dict(d):
    G = nx.Graph()
    G.add_nodes_from((n, a) for n, a in d["nodes"])
    G.add_edges_from((u, v, a) for u, v, a in d["edges"])
    return G


class Retriever:
    """Hybrid: embedding (sentence-transformers) + TF-IDF.

    Không cài được sentence-transformers thì chỉ dùng TF-IDF."""

    def __init__(self, texts, use_embeddings=True, model_name="all-MiniLM-L6-v2"):
        self.texts, self.mode = texts, "tfidf"
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                try:
                    from transformers.utils import logging as hf_log

                    hf_log.disable_progress_bar()
                    hf_log.set_verbosity_error()
                except Exception:
                    pass
                self.model = SentenceTransformer(model_name)
                self.emb = self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=64,
                    show_progress_bar=False,
                )
                self.mode = "hybrid"
            except Exception as e:
                print(
                    "Không dùng được embedding, chuyển sang TF-IDF:", type(e).__name__
                )
        self.vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.tfidf = self.vec.fit_transform(texts)

    def scores(self, q):
        lex = (self.tfidf @ self.vec.transform([q]).T).toarray().ravel()
        if self.mode == "hybrid":
            return (
                self.emb @ self.model.encode([q], normalize_embeddings=True)[0]
                + 0.5 * lex
            )
        return lex


class GraphRAG:
    """RAG thường (mode='rag') hoặc RAG + KG (mode='graph')."""

    def __init__(self, docs, lex, G, retriever, hub_ratio=0.3):
        self.docs, self.G, self.ret = docs, G, retriever
        self.rx = compile_lexicon(lex)
        freq = collections.Counter(e for d in docs for e in d["entities"])
        self.hubs = {e for e, c in freq.items() if c / len(docs) > hub_ratio}

    def entities(self, text):
        return extract_entities(text, self.rx)

    def _ent_neighbors(self, e):
        return [
            (nb, d)
            for nb, d in self.G[e].items()
            if self.G.nodes[nb].get("kind") == "entity"
        ]

    def neighbors(self, ents, topn=4):
        cand = collections.Counter()
        for e in ents:
            if e not in self.G:
                continue
            for nb, d in self._ent_neighbors(e):
                if nb not in ents and nb not in self.hubs:
                    cand[nb] += d.get("weight", 1)
        return [n for n, _ in cand.most_common(topn)]

    def facts(self, ents, per_entity=4):
        out = []
        for e in ents:
            if e not in self.G:
                continue
            for nb, d in sorted(
                self._ent_neighbors(e), key=lambda x: -x[1].get("weight", 1)
            )[:per_entity]:
                extra = f", corr={d['corr']:+.2f}" if "corr" in d else ""
                rel, w = d.get("rel", "related"), d.get("weight", 1)
                out.append(f"({e}) -[{rel}, w={w:.1f}{extra}]-> ({nb})")
        return out

    def retrieve(self, q, k=4, mode="graph", alpha=0.10, beta=0.04):
        s = np.array(self.ret.scores(q), dtype=float)
        qe = self.entities(q)
        nb = self.neighbors(qe) if mode == "graph" else []
        if mode == "graph":
            for i, d in enumerate(self.docs):
                de = set(d["entities"])
                s[i] += alpha * len(de & set(qe)) + beta * len(de & set(nb))
        idx = np.argsort(-s)[:k]
        return (
            [
                {
                    "i": int(i),
                    "score": float(s[i]),
                    "entities": self.docs[i]["entities"],
                }
                for i in idx
            ],
            qe,
            nb,
        )

    def answer(self, q, k=4, mode="graph", n_sent=3):
        hits, qe, nb = self.retrieve(q, k, mode)
        qw = set(re.findall(r"[a-z0-9\-]{3,}", q.lower())) - STOP
        cands, seen = [], set()
        for h in hits:
            for sent in split_sentences(self.docs[h["i"]]["text"]):
                if sent in seen:
                    continue
                seen.add(sent)
                ov = len(qw & set(re.findall(r"[a-z0-9\-]{3,}", sent.lower())))
                cands.append(
                    (ov, h["score"], sent, self.docs[h["i"]].get("source", ""))
                )
        cands.sort(key=lambda x: (-x[0], -x[1]))
        top = cands[:n_sent]
        return {
            "answer": " ".join(c[2] for c in top),
            "sources": sorted({c[3] for c in top if c[3]}),
            "query_entities": qe,
            "graph_expansion": nb,
            "kg_facts": self.facts(qe) if mode == "graph" else [],
            "hits": hits,
        }


def context_recall(rag, questions, k=4, mode="graph"):
    """Tỉ lệ câu hỏi mà top-k ngữ cảnh chứa đủ mọi từ khóa vàng (regex)."""
    ok = []
    for q, gold in questions:
        hits, _, _ = rag.retrieve(q, k, mode)
        ctx = " ".join(rag.docs[h["i"]]["text"] for h in hits).lower()
        ok.append(all(re.search(g, ctx) for g in gold))
    return float(np.mean(ok)), ok
