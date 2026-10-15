import json, os
import gradio as gr
from ragkg_core import GraphRAG, Retriever, graph_from_dict

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = {"Tiểu đường": "Which drugs are studied for diabetic kidney disease?",
            "Chứng khoán": "How did Hoa Phat perform in 2021?",
            "Vàng": "How do Fed rate cuts affect gold prices?"}
FILES = [("Tiểu đường", "diabetes.json"), ("Chứng khoán", "stocks.json"),
         ("Vàng", "gold.json")]
SYSTEMS = {}
for name, f in FILES:
    d = json.load(open(os.path.join(HERE, f), encoding="utf-8"))
    retr = Retriever([x["text"] for x in d["docs"]], use_embeddings=False)
    G = graph_from_dict(d["graph"])
    SYSTEMS[name] = GraphRAG(d["docs"], d["lexicon"], G, retr)


def ask(domain, question, mode, k):
    m = "graph" if mode == "GraphRAG" else "rag"
    r = SYSTEMS[domain].answer(question, k=int(k), mode=m)
    facts = "\n".join(r["kg_facts"]) or "(RAG thường: không dùng KG)"
    expand = (", ".join(r["query_entities"]) + " -> "
              + ", ".join(r["graph_expansion"]))
    return r["answer"], ", ".join(r["sources"]), expand, facts


SLUG = {"diabetes": "Tiểu đường", "stocks": "Chứng khoán", "gold": "Vàng"}


def from_url(request: gr.Request):
    # ?domain=gold&q=...&mode=rag: mở sẵn câu hỏi và trả lời ngay khi tải trang
    p = dict(request.query_params) if request else {}
    d = SLUG.get(p.get("domain"), "Tiểu đường")
    q = p.get("q") or EXAMPLES[d]
    m = "RAG" if p.get("mode") == "rag" else "GraphRAG"
    return (d, q, m, *ask(d, q, m, 4))


with gr.Blocks(title="KG + RAG") as demo:
    gr.Markdown("# Knowledge Graph + RAG: tiểu đường, chứng khoán, vàng\n"
                "Hỏi bằng tiếng Anh (kho dữ liệu là tiếng Anh).")
    domain = gr.Dropdown(list(SYSTEMS), value="Tiểu đường", label="Bài")
    question = gr.Textbox(label="Câu hỏi", value=EXAMPLES["Tiểu đường"])
    domain.input(lambda d: EXAMPLES[d], domain, question)
    with gr.Row():
        mode = gr.Radio(["GraphRAG", "RAG"], value="GraphRAG", label="Chế độ")
        k = gr.Slider(1, 8, value=4, step=1, label="top-k")
    btn = gr.Button("Hỏi")
    ans = gr.Textbox(label="Câu trả lời", lines=6)
    src = gr.Textbox(label="Nguồn")
    ent = gr.Textbox(label="Thực thể -> mở rộng KG")
    facts = gr.Textbox(label="Facts từ KG", lines=5)
    btn.click(ask, [domain, question, mode, k], [ans, src, ent, facts])
    demo.load(from_url, None, [domain, question, mode, ans, src, ent, facts])


if __name__ == "__main__":
    demo.launch()
