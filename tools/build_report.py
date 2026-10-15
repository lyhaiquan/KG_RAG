"""Sinh bao cao bai 07: A05_07_quanlh.676.md -> .docx (pandoc) -> .pdf (Word).

Code trong bao cao duoc trich tu chinh file nguon (deploy/ragkg_core.py va notebook)
qua report_lib, output trich tu notebook da chay, con cac con so trong van ban doc tu
results/summary.json do notebook ghi ra. Nho vay bao cao, notebook va ket qua khop nhau.

Chay sau khi da chay xong notebook:  python tools/build_report.py [--no-pdf]
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import report_lib as R                                    # noqa: E402

NB = "RAG_KG_Kaggle.ipynb"
CORE = "deploy/ragkg_core.py"
NAME = "A05_07_quanlh.676"
OUT_MD = ROOT / (NAME + ".md")
OUT_DOCX = ROOT / (NAME + ".docx")
OUT_PDF = ROOT / (NAME + ".pdf")
REPO = "https://github.com/lyhaiquan/KG_RAG"

RES = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
EV = RES["eval"]
DB, ST, GD = RES["diabetes"], RES["stocks"], RES["gold"]


# ------------------------------------------------------------------ tien ich

def vn(x, nd=3):
    """So thap phan kieu Viet Nam: dau phay."""
    return ("%.*f" % (nd, x)).replace(".", ",")


def pct(x, nd=1, sign=False):
    s = ("%+.*f" if sign else "%.*f") % (nd, 100 * x)
    return s.replace(".", ",") + "%"


def num(n):
    return "{:,}".format(n).replace(",", ".")


def ev(dom, k, m):
    return EV[dom][str(k)][m]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def check(cond, msg):
    """Moi nhan dinh trong van ban phai dung voi ket qua lan chay cuoi."""
    if not cond:
        raise SystemExit("NHAN DINH KHONG CON DUNG: " + msg)


FIG_NO = [0]


def fig(rel, caption, width="100%"):
    FIG_NO[0] += 1
    return "![Hình %d. %s](%s){width=%s}\n" % (FIG_NO[0], caption, rel, width)


MD = []


def w(*chunks):
    for c in chunks:
        MD.append(c.rstrip("\n"))
        MD.append("")


def core(*names):
    w(R.py(textwrap.dedent(R.obj(CORE, *names))))


def nbdef(needle, *names):
    w(R.py(R.nbobj(NB, needle, *names)))


def nbcode(needle, first, last):
    w(R.py(R.nbstmt(NB, needle, first, last)))


def output(needle, max_lines=None, squeeze=False):
    text = R.nbout(NB, needle, max_lines)
    if squeeze:
        text = "\n".join(" ".join(l.split()) for l in text.splitlines())
    w(R.out(text))


def eval_fig():
    """Bieu do tong hop context recall cua 3 bai, ve tu results/summary.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doms = [("diabetes", "Tiểu đường"), ("stocks", "Chứng khoán"), ("gold", "Vàng")]
    fig_, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (d, title) in zip(axes, doms):
        ks = sorted(EV[d], key=int)
        xs = range(len(ks))
        for j, (m, c) in enumerate([("RAG", "#b8b8b0"), ("GraphRAG", "#2f6fd0")]):
            vals = [EV[d][k][m] for k in ks]
            bars = ax.bar([x + (j - 0.5) * 0.38 for x in xs], vals, 0.36, color=c, label=m)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, vn(v, 2),
                        ha="center", fontsize=8)
        ax.set_xticks(list(xs), ["k=" + k for k in ks])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("context recall@k")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(ROOT / "figs" / "eval_summary.png", dpi=170)
    plt.close()


# ================================================================== 1. GIOI THIEU
w("""---
title: "Báo cáo bài tập 07: Knowledge Graph và RAG"
subtitle: "Xây dựng Knowledge Graph và hệ hỏi đáp RAG, GraphRAG cho bệnh tiểu đường, chứng khoán và vàng"
author: "Lý Hải Quân - B23DCCN676"
date: "Học phần Phát triển hệ thống thông minh"
lang: vi
toc-title: "Mục lục"
---""")

w("# 1. Giới thiệu")
w("## 1.1. Nội dung bài tập")
w("""Đề bài là chương 4 "Knowledge Graph và RAG" với sáu mục: mở đầu về lịch sử và vai trò
của KG và RAG, Knowledge Graph và Knowledge Base, RAG, rồi áp dụng KB và RAG cho ba bài
toán (bệnh tiểu đường, chứng khoán, vàng) và triển khai (deploy). Toàn bộ phần code nằm
trong một notebook chạy được trên Kaggle; mỗi mục của đề ứng với một chương của báo cáo:""")

w(table(["Mục trên đề", "Nội dung", "Chương"],
        [["4.1", "Mở đầu: KG và RAG, lịch sử; vai trò, ý nghĩa với AI", "2"],
         ["4.2", "Knowledge Graph và Knowledge Base", "3"],
         ["4.3", "RAG (và GraphRAG)", "4"],
         ["4.4", "KB và RAG cho bệnh tiểu đường", "5"],
         ["4.5", "KB và RAG cho chứng khoán, vàng", "6, 7"],
         ["4.6", "Deploy", "8"],
         ["", "So sánh, đánh giá ba bài", "9"]]))

w("## 1.2. Ba bộ dữ liệu")
w(table(["", "Tiểu đường", "Chứng khoán", "Vàng"],
        [["Nguồn", "PubMed (NCBI E-utilities)", "yfinance, Google News RSS",
          "yfinance, Google News RSS"],
         ["Văn bản (text)", "%s abstract bài báo y khoa" % num(DB["papers"]),
          "%s tiêu đề tin + mô tả 5 công ty" % num(ST["news"]),
          "%s tiêu đề tin về vàng" % num(GD["news"])],
         ["Bảng (table)", "nhãn MeSH của từng bài",
          "giá đóng cửa VNM, FPT, VCB, HPG, VIC; ngành, lĩnh vực",
          "giá vàng và 7 yếu tố: USD, lợi suất TP Mỹ 10 năm, bạc, dầu, S&P 500, VIX, Bitcoin"],
         ["Giai đoạn", "kết quả liên quan nhất của truy vấn",
          "%s đến %s (%s phiên)" % (ST["start"], ST["end"], num(ST["rows"])),
          "cùng giai đoạn (%s dòng)" % num(GD["rows"])],
         ["Câu hỏi đánh giá", str(RES["n_questions"]["diabetes"]),
          str(RES["n_questions"]["stocks"]), str(RES["n_questions"]["gold"])]]))

w("""Ba bộ dữ liệu được chọn để có đủ hai dạng dữ liệu mà một hệ hỏi đáp thực tế gặp phải.
Bộ tiểu đường gần như thuần văn bản: tri thức nằm trong các câu của bài báo, và phải trích
thực thể (thuốc, biến chứng, bệnh) ra mới dựng được đồ thị. Hai bộ tài chính ngược lại:
phần lớn tri thức nằm trong **bảng số** (giá theo ngày), thứ mà RAG không đọc trực tiếp
được, cộng với tin tức dạng văn bản ngắn. Bộ chứng khoán dùng năm mã lớn của sàn HOSE ở
năm ngành khác nhau; bộ vàng đặt giá vàng cạnh các yếu tố vĩ mô thường được nhắc tới khi
giải thích giá vàng. Dữ liệu được tải lại mỗi lần chạy notebook; thư mục `data/` giữ bản
chụp ngày 15/10/2026 mà báo cáo này dùng.""")

w("## 1.3. Môi trường và cấu trúc thư mục")
w("""Notebook được viết để chạy trên Kaggle (bật Internet, CPU là đủ) và cũng chạy được trên
máy cá nhân: các số liệu trong báo cáo lấy từ lần chạy trên máy cá nhân với Python 3.14,
CPU. Mô hình embedding là `all-MiniLM-L6-v2` của sentence-transformers (khoảng 90 MB, tải
lần đầu từ Hugging Face); đồ thị dùng NetworkX; truy xuất từ khóa dùng TF-IDF của
scikit-learn; giao diện dùng Gradio. Không có thành phần nào cần GPU hay API key trả phí.""")

w("""```
KG_RAG/
  RAG_KG_Kaggle.ipynb        notebook chính, mục 4.1-4.6, tự tải dữ liệu nếu thiếu
  data/                      bản chụp dữ liệu: PubMed, giá, tin tức (15/10/2026)
  results/summary.json       số liệu chính do notebook ghi ra
  figs/                      hình do notebook vẽ
  deploy/
    ragkg_core.py            phần lõi: chunking, NER từ điển, dựng KG, retriever, GraphRAG
    app.py                   giao diện Gradio cho cả 3 bài
    diabetes.json, stocks.json, gold.json    KB + KG + từ điển đã đóng gói
  docs/demo/                 ảnh chụp giao diện
  tools/build_report.py      sinh báo cáo này (.md -> .docx -> .pdf)
  A05_07_quanlh.676.docx / .pdf   báo cáo này
  README.md
```""")

w("""File `ragkg_core.py` được chính notebook ghi ra (lệnh `%%writefile`) rồi import lại, nên
notebook và bản deploy dùng chung đúng một bản code. Mọi đoạn code trong báo cáo được trích
tự động từ file này và từ notebook, không gõ lại bằng tay. Mã nguồn đầy đủ ở """ + REPO + ".")

# ================================================================== 2. MO DAU
w("# 2. Mở đầu: KG và RAG (mục 4.1)")
w("## 2.1. Lịch sử Knowledge Graph")
w("""Ý tưởng biểu diễn tri thức bằng nút và cạnh có từ thập niên 1960 với *semantic network*
của Quillian: mỗi khái niệm là một nút, quan hệ giữa các khái niệm là cạnh có nhãn. Năm
2001, Berners-Lee đề xuất *Semantic Web*, chuẩn hóa tri thức thành các bộ ba (chủ thể, quan
hệ, đối tượng) với RDF, OWL và ngôn ngữ truy vấn SPARQL. Giai đoạn 2007-2014 xuất hiện các
KG mở quy mô lớn như DBpedia, Freebase, Wikidata. Năm 2012 Google công bố Knowledge Graph
với khẩu hiệu "things, not strings": máy tìm kiếm trả lời về *thực thể* chứ không chỉ khớp
chuỗi ký tự. Từ đó KG thành hạ tầng cho tìm kiếm, hệ gợi ý, y sinh (UMLS, DrugBank,
Hetionet) và tài chính.""")

w("## 2.2. Lịch sử RAG")
w("""RAG (Retrieval-Augmented Generation) được Lewis và cộng sự (Facebook AI) đề xuất năm 2020:
trước khi sinh câu trả lời, mô hình truy xuất các đoạn văn liên quan từ một kho tài liệu
và sinh câu trả lời dựa trên các đoạn đó. Giai đoạn 2022-2023, khi các mô hình ngôn ngữ lớn
(LLM) phổ biến, RAG kết hợp với cơ sở dữ liệu vector (FAISS, Chroma...) thành mẫu thiết kế
tiêu chuẩn cho chatbot trên tài liệu riêng. Năm 2024, GraphRAG (Microsoft và cộng đồng)
đưa KG vào RAG: thay vì chỉ tìm đoạn văn giống câu hỏi, hệ thống đi theo quan hệ trong đồ
thị để gom thông tin nằm rải rác ở nhiều tài liệu.""")

w("## 2.3. Vai trò và ý nghĩa với AI")
w(table(["Vấn đề của LLM thuần", "KG / RAG giải quyết thế nào"],
        [["Bịa thông tin (hallucination)", "câu trả lời dựa trên tài liệu truy xuất được, có trích nguồn"],
         ["Kiến thức cũ", "chỉ cần cập nhật kho tài liệu, không phải huấn luyện lại"],
         ["Khó giải thích", "KG cho thấy đường suy luận: thực thể, quan hệ, thực thể"],
         ["Câu hỏi nhiều bước", "duyệt đồ thị để nối thông tin nằm ở nhiều tài liệu"],
         ["Dữ liệu chuyên ngành, riêng tư", "dùng kho tri thức nội bộ, không phụ thuộc dữ liệu huấn luyện"]]))
w("""Hai kỹ thuật bổ sung cho nhau. RAG mạnh ở chỗ tìm đúng đoạn văn có câu trả lời; KG
mạnh ở chỗ biết các thực thể liên quan với nhau thế nào, kể cả khi chúng không xuất hiện
trong cùng một đoạn văn. Bài này đo trực tiếp khi nào phần KG giúp ích, bằng cách chạy cùng
một bộ câu hỏi qua RAG thường và GraphRAG (chương 9).""")

# ================================================================== 3. KG VA KB
w("# 3. Knowledge Graph và Knowledge Base (mục 4.2)")
w("## 3.1. Khái niệm")
w("""**Knowledge Base (KB)** là kho tri thức có cấu trúc hoặc bán cấu trúc: các đoạn văn đã
làm sạch, bảng số liệu, ontology. **Knowledge Graph (KG)** là KB được biểu diễn dưới dạng
đồ thị, gồm các bộ ba (chủ thể, quan hệ, đối tượng), ví dụ
(SGLT2 inhibitor, drug_studied_for, Nephropathy). Trong bài, KB của mỗi bài toán là một
danh sách *tài liệu*, mỗi tài liệu có ba trường: nội dung văn bản, nguồn (PMID, mã tin, mã
bảng thống kê) và danh sách thực thể xuất hiện trong nó. KG được dựng từ KB đó.""")

w("## 3.2. Từ bảng và văn bản tới KB")
w("""Văn bản dài (abstract) được chia thành đoạn ngắn để truy xuất chính xác hơn; bảng được
chuyển thành câu văn vì RAG chỉ truy xuất được văn bản. Hai hàm dưới cắt văn bản thành câu
và gom 3 câu thành một đoạn, có gắn tiêu đề bài ở đầu để đoạn nào cũng giữ được ngữ cảnh:""")
core("split_sentences", "chunk_text")
w("""Chuyển bảng thành câu văn là việc của chương 6 và 7 (hàm `stats_to_text`, `corr_text`).""")

w("## 3.3. Nhận diện thực thể")
w("""Thực thể được nhận diện bằng từ điển: mỗi thực thể có một loại và một biểu thức chính
quy gom các cách viết khác nhau về một tên chuẩn, ví dụ "empagliflozin", "dapagliflozin"
đều là `SGLT2 inhibitor`.""")
core("compile_lexicon", "extract_entities")
w("""`compile_lexicon` biên dịch trước mọi regex với cờ `re.I` (không phân biệt hoa thường),
vì cùng một regex sẽ được dùng cho hàng nghìn đoạn văn. `extract_entities` giữ lại thực thể
nào có regex tìm thấy trong đoạn văn và sắp xếp kết quả để mỗi lần chạy ra cùng thứ tự.
So với dùng mô hình NER hay LLM, cách này chính xác, nhanh, không cần API và dễ kiểm tra
bằng mắt, đổi lại độ phủ thấp hơn (chỉ bắt được cách viết có trong từ điển).""")

w("## 3.4. Dựng Knowledge Graph")
w("""Quan hệ được lấy từ ba nguồn. Từ **văn bản**: hai thực thể cùng xuất hiện trong một
tài liệu thì nối một cạnh. Từ **bảng**: hệ số tương quan lợi suất thành cạnh
`CORRELATED_WITH`, ngành của công ty thành cạnh `BELONGS_TO_SECTOR` (chương 6, 7). Từ **tri
thức chuyên gia**: vài quan hệ nhân quả kinh tế đã biết (bài vàng). Phần dùng chung:""")
core("make_rel", "new_kg", "add_cooccurrence")
w("""`make_rel` nhận một bảng tra (cặp kiểu thực thể -> tên quan hệ) và trả về hàm đặt tên
quan hệ không phụ thuộc thứ tự hai đầu: (Drug, Complication) và (Complication, Drug) đều
cho `drug_studied_for`. `new_kg` tạo trước đủ các nút thực thể của từ điển, kể cả thực thể
chưa xuất hiện, để đồ thị của mọi lần chạy có cùng tập nút.

`add_cooccurrence` duyệt từng tài liệu. Thực thể loại `Year` bị bỏ vì năm xuất hiện cùng
mọi thứ và chỉ tạo cạnh nhiễu. Nếu có `doc_key` (bài tiểu đường dùng PMID), hàm thêm một nút
cho tài liệu và cạnh `MENTIONS` tới từng thực thể, nhờ đó từ một thực thể có thể lần ngược
ra các bài báo nhắc tới nó. `itertools.combinations(ents, 2)` sinh mọi cặp thực thể trong
tài liệu; cặp đã có cạnh thì tăng `weight` thêm 1, nên trọng số cuối cùng là số tài liệu mà
hai thực thể cùng xuất hiện. Thuộc tính `source="text"` ghi lại nguồn gốc của cạnh để phân
biệt với cạnh từ bảng (`table`) và từ tri thức chuyên gia (`domain`).""")

w("## 3.5. Lưu trữ đồ thị")
core("graph_to_dict", "graph_from_dict")
w("""Đồ thị được lưu dưới dạng danh sách nút và cạnh kèm thuộc tính, ghi vào JSON cùng KB và
từ điển để bản deploy nạp lại mà không phải chạy lại notebook (chương 8). Notebook cũng xuất
GraphML (`nx.write_graphml`) để có thể nạp vào Neo4j khi cần truy vấn bằng Cypher.""")

# ================================================================== 4. RAG
w("# 4. RAG và GraphRAG (mục 4.3)")
w("## 4.1. Kiến trúc")
w(fig("figs/pipeline.png",
      "Pipeline dùng chung cho ba bài. RAG thường chỉ đi đường dưới; GraphRAG thêm nhánh "
      "trên: mở rộng thực thể của câu hỏi qua KG rồi cộng điểm cho tài liệu chứa chúng"))
w("""Cả ba bài dùng cùng một pipeline. Câu hỏi được chấm điểm với mọi tài liệu trong KB bằng
truy xuất vector; GraphRAG cộng thêm điểm dựa trên KG; k tài liệu điểm cao nhất được dùng để
dựng câu trả lời. Điểm của tài liệu $d$ với câu hỏi $q$:""")
w(r"""$$\mathrm{score}(d) = \cos_{\mathrm{emb}}(q,d) + 0{,}5\,\cos_{\mathrm{tfidf}}(q,d) + \alpha\,|E(q)\cap E(d)| + \beta\,|N(q)\cap E(d)|$$""")
w("""trong đó $E(q)$, $E(d)$ là tập thực thể của câu hỏi và của tài liệu, $N(q)$ là các
thực thể láng giềng của $E(q)$ trong KG, $\\alpha = 0{,}10$ và $\\beta = 0{,}04$. RAG thường
là trường hợp $\\alpha = \\beta = 0$.""")

w("## 4.2. Truy xuất hybrid")
core("Retriever")
w("""TF-IDF luôn được dựng. `self.vec.transform([q])` biến câu hỏi thành vectơ TF-IDF; nhân
với ma trận TF-IDF của toàn bộ tài liệu cho ra cosine với từng tài liệu, vì scikit-learn đã
chuẩn hóa mỗi vectơ về độ dài 1. Nếu cài được sentence-transformers, câu hỏi và tài liệu
được mã hóa bằng mô hình embedding với `normalize_embeddings=True`, nên tích vô hướng cũng là
cosine. Điểm cuối = cosine embedding + 0,5 × cosine TF-IDF: phần ngữ nghĩa làm chính, phần
từ khóa bù cho những thứ embedding hay bỏ lỡ như mã cổ phiếu "HPG" hay năm "2021". Không có
sentence-transformers thì chỉ dùng TF-IDF, nên code vẫn chạy được; bản deploy chủ động dùng
cách này cho nhẹ.""")

w("## 4.3. GraphRAG: mở rộng qua KG và cộng điểm")
core("neighbors", "retrieve")
w("""`neighbors` duyệt các nút thực thể kề với mỗi thực thể trong câu hỏi, bỏ những láng giềng
đã có trong câu hỏi và các nút *hub*: thực thể có mặt trong hơn 30% tài liệu (ví dụ GOLD
trong bài vàng). Hub nối với gần như mọi thứ nên không giúp phân biệt tài liệu. Trọng số cạnh
được cộng dồn vào `Counter`, nên một láng giềng nối với nhiều thực thể của câu hỏi được xếp
cao hơn; hàm trả về tối đa 4 láng giềng, chính là $N(q)$.

`retrieve` tính điểm vector cho mọi tài liệu, nhận diện $E(q)$ bằng cùng từ điển đã dùng
cho tài liệu, rồi ở chế độ `graph` cộng $\\alpha$ cho mỗi thực thể trùng với câu hỏi và
$\\beta$ cho mỗi thực thể trùng với láng giềng. `np.argsort(-s)[:k]` lấy k tài liệu điểm
cao nhất. Hàm trả kèm $E(q)$ và $N(q)$ để giao diện hiển thị phần mở rộng KG.""")

w("## 4.4. Sinh câu trả lời")
core("facts", "answer")
w("""Câu trả lời mặc định là *trích xuất*: tách top-k tài liệu thành câu, chấm mỗi câu theo
số từ trùng với câu hỏi (đã bỏ stopword), lấy 3 câu cao nhất, kèm nguồn. Ở chế độ GraphRAG,
`facts` đọc các cạnh mạnh nhất quanh thực thể của câu hỏi và in thành bộ ba có trọng số,
ví dụ `(HPG) -[CORRELATED_WITH, w=5.3, corr=+0.43]-> (VCB)`. Cách trích xuất không cần API
key và không thể bịa thông tin; notebook có thêm một ô tùy chọn dùng LLM flan-t5-large để
viết lại câu trả lời từ cùng ngữ cảnh và facts.""")

w("## 4.5. Cách đánh giá")
core("context_recall")
w("""Mỗi câu hỏi có một danh sách từ khóa vàng dạng regex. Câu hỏi được tính đúng khi top-k
tài liệu truy xuất được chứa **đủ** mọi từ khóa. Ví dụ câu "How do SGLT2 inhibitors affect
kidney disease in diabetes?" có hai regex `sglt|gliflozin` và `kidney|nephro`. Đây là độ đo
cho bước truy xuất (context recall@k), không đo chất lượng câu văn trả lời; nó trả lời câu
hỏi chính của bài: phần KG có giúp tìm đủ ngữ cảnh hơn không. Mỗi bài có 10-12 câu hỏi, đánh
giá ở k = 2, 4, 6.""")

# ================================================================== 5. TIEU DUONG
w("# 5. Bài 1: KB và RAG cho bệnh tiểu đường (mục 4.4)")
w("## 5.1. Dữ liệu")
w("""Dữ liệu là %s abstract PubMed lấy qua NCBI E-utilities, truy vấn theo MeSH "diabetes
mellitus" kèm treatment, complications hoặc diagnosis, sắp theo mức liên quan:""" % num(DB["papers"]))
nbdef("def fetch_pubmed", "fetch_pubmed")
w("""`esearch` trả về danh sách PMID, `efetch` lấy chi tiết theo lô 100 bài. Mỗi bài giữ lại
tiêu đề, abstract (ghép các phần Background, Methods...) và danh sách nhãn MeSH.""")
output("PUBMED_FILE), encoding")

w("## 5.2. Knowledge Base")
nbcode("DIAB_LEX = {", "DIAB_LEX = {", "diab_docs = [")
output_note = ("Từ điển có %d thực thể thuộc ba loại Drug (8), Complication (8), Disease (5). "
               "Các regex có cả vài cách viết tiếng Việt (\"tiểu đường type 2\", \"béo phì\") "
               "để câu hỏi tiếng Việt vẫn nhận diện được thực thể. KB gồm %s đoạn, trung bình "
               "%s thực thể mỗi đoạn." % (DB["entities"], num(DB["chunks"]), vn(DB["ent_per_chunk"], 2)))
w(output_note)

w("## 5.3. Knowledge Graph")
nbcode("diab_rel = make_rel", "diab_rel = make_rel", "G_diab = add_cooccurrence")
output("diab_rel = make_rel")
w(fig("figs/kg_diabetes.png",
      "KG tiểu đường, chỉ vẽ nút thực thể và cạnh đồng xuất hiện từ 2 đoạn trở lên. "
      "Màu theo loại, độ dày cạnh theo trọng số"))
w("""KG có %s nút (%d thực thể và %s bài báo) và %s cạnh. Các bộ ba mạnh nhất cho thấy KG
học được những liên hệ quen thuộc trong y văn: insulin được nghiên cứu cùng bệnh tim mạch,
bệnh thận là biến chứng của đái tháo đường type 2, GLP-1 RA đi cùng type 2 và béo phì. Vì
quan hệ dựa trên đồng xuất hiện, cạnh chỉ nói hai thực thể được nhắc cùng nhau, không khẳng
định thuốc nào điều trị bệnh nào.""" % (num(DB["nodes"]), DB["entities"], num(DB["papers"]), num(DB["edges"])))

w("## 5.4. Hỏi đáp mẫu và đánh giá")
output("diab_rag = GraphRAG", max_lines=40)
w(fig("figs/eval_diabetes.png", "Context recall@k của RAG thường và GraphRAG, bài tiểu đường",
      width="55%"))
d2r, d2g = ev("diabetes", 2, "RAG"), ev("diabetes", 2, "GraphRAG")
d4r, d4g = ev("diabetes", 4, "RAG"), ev("diabetes", 4, "GraphRAG")
check(d2g > d2r and d4g >= d4r, "GraphRAG hon RAG o bai tieu duong")
w("""Với câu hỏi mẫu, RAG thường lấy được các bài về bệnh thận do đái tháo đường nhưng bỏ
qua khía cạnh tim mạch. GraphRAG nhận diện Nephropathy và Cardiovascular disease, mở rộng
sang các láng giềng mạnh trong KG và kéo thêm bài nói về nguy cơ xơ vữa, tim mạch ở bệnh
nhân bệnh thận mạn. Trên cả bộ câu hỏi, GraphRAG tăng context recall@2 từ %s lên %s và
recall@4 từ %s lên %s. Đây là bài GraphRAG có lợi rõ nhất, vì câu hỏi y khoa thường nối nhiều
thực thể (thuốc với biến chứng) mà một đoạn văn đơn lẻ không chứa đủ.""" % (vn(d2r), vn(d2g), vn(d4r), vn(d4g)))

# ================================================================== 6. CHUNG KHOAN
w("# 6. Bài 2: KB và RAG cho chứng khoán (mục 4.5)")
w("## 6.1. Dữ liệu và phân tích nhanh")
w("""Giá đóng cửa đã điều chỉnh của năm cổ phiếu lấy từ yfinance (hậu tố `.VN`), thông tin
công ty từ `Ticker.info`, tin tức từ Google News RSS với tên từng công ty làm từ khóa:""")
nbdef("def fetch_news", "fetch_news")
w("""Kết quả RSS là XML; `findall(".//item")` lấy từng tin, tiêu đề có dạng "Nội dung - Tên
báo" nên `rsplit(" - ", 1)[0]` cắt bỏ tên báo, và `drop_duplicates` bỏ tin trùng giữa các
từ khóa. Hai hàm chuyển bảng giá thành số liệu theo năm:""")
nbdef("def yearly_stats", "yearly_stats", "weekly_changes")
w("""Lợi suất năm = giá cuối năm / giá đầu năm - 1; độ biến động năm hóa = độ lệch chuẩn lợi
suất ngày × √252; sụt giảm tối đa dùng `g.cummax()` (đỉnh cao nhất tới từng ngày) và lấy giá
trị nhỏ nhất của `g / g.cummax() - 1`. Tương quan được tính trên lợi suất **tuần** để giảm
nhiễu và tránh lệch ngày nghỉ; với chỉ số dạng mức (lợi suất trái phiếu, VIX) dùng chênh lệch
mức thay cho phần trăm.""")
output("sp = pd.read_csv", max_lines=30, squeeze=True)
w(fig("figs/stock_prices.png", "Giá chuẩn hóa của năm cổ phiếu (đầu kỳ = 1)"))
bests = ST["best_year"]
w("""Bốn mã VNM, FPT, VCB, HPG tương quan với nhau ở mức 0,3-0,4; VIC gần như đi riêng. Năm
lợi suất cao nhất của từng mã: %s. Mức tăng của VIC năm 2025 là số liệu của yfinance và
được giữ nguyên.""" % "; ".join("%s %s (%s)" % (a, pct(v[1], 0, True), v[0]) for a, v in sorted(bests.items())))

w("## 6.2. Knowledge Base: chuyển bảng thành văn bản")
nbdef("def stats_to_text", "stats_to_text", "corr_text")
w("""Mỗi dòng thống kê thành một câu tiếng Anh, ví dụ "HPG (Hoa Phat Group..., sector Basic
Materials) in 2021: price return +52.7%, annualized volatility 34.7%...". `{r['return']:+.1%}`
định dạng số thành phần trăm có dấu. Câu chứa mã, tên công ty, ngành và năm, nên cả TF-IDF lẫn
từ điển thực thể đều bắt được. Từ điển của bài có công ty, ngành, chủ đề tin (lợi nhuận, cổ
tức, khối ngoại, VN-Index, xuất khẩu, AI và chip, xe điện) và các **năm** 2020-2026: năm là
thực thể để câu hỏi "năm 2021" được nối với đúng bản ghi thống kê.""")
output("STOCK_LEX = {", squeeze=True)

w("## 6.3. Knowledge Graph")
nbcode("G_stock = new_kg", "G_stock = new_kg", "add_cooccurrence(G_stock")
w("""Cạnh `BELONGS_TO_SECTOR` lấy từ thông tin công ty, ngành chi tiết lưu làm thuộc tính.
Chỉ cặp có |corr| từ 0,15 trở lên mới thành cạnh `CORRELATED_WITH`; trọng số = 10 × |corr| để
cùng thang với cạnh đếm từ tin tức, dấu thật giữ trong thuộc tính `corr`. Cuối cùng
`add_cooccurrence` trên tin tức thêm cạnh công ty - chủ đề.""")
output("G_stock = new_kg", squeeze=True)
w(fig("figs/kg_stocks.png", "KG chứng khoán: ngành và tương quan từ bảng, chủ đề từ tin tức"))

w("## 6.4. Hỏi đáp mẫu và đánh giá")
output("stock_rag = GraphRAG", max_lines=40)
w(fig("figs/eval_stocks.png", "Context recall@k, bài chứng khoán", width="55%"))
s4r, s4g = ev("stocks", 4, "RAG"), ev("stocks", 4, "GraphRAG")
check(s4g > s4r, "GraphRAG hon RAG o bai chung khoan, k=4")
w("""Với câu hỏi về Hòa Phát năm 2021, cả hai chế độ đều tìm được bản ghi thống kê và tin
tức, nhưng GraphRAG trả kèm facts về ngành và tương quan của HPG. Trên cả bộ câu hỏi,
GraphRAG nâng recall@4 từ %s lên %s: những câu hỏi ghép nhiều thực thể (công ty + chủ đề tin,
hai công ty + một năm) được lợi từ điểm cộng theo thực thể.""" % (vn(s4r), vn(s4g)))

# ================================================================== 7. VANG
w("# 7. Bài 3: KB và RAG cho vàng (mục 4.5)")
w("## 7.1. Dữ liệu và phân tích nhanh")
output("gm = pd.read_csv", max_lines=30, squeeze=True)
w(fig("figs/gold_price_corr.png",
      "Trái: giá vàng (USD/oz). Phải: tương quan trượt 26 tuần giữa vàng và USD, lợi suất "
      "trái phiếu 10 năm, bạc"))
gr_ = GD["gold_return"]
cg = GD["corr_gold"]
check(cg["USD"] < 0 and cg["SILVER"] > 0.5, "vang nguoc USD, cung bac")
w("""Vàng tăng mạnh nhất năm 2025 (%s), sau %s năm 2024. Tương quan tuần với vàng: bạc %s,
USD index %s, lợi suất trái phiếu Mỹ 10 năm %s. Đồ thị tương quan trượt cho thấy quan hệ
âm với USD và lợi suất không cố định mà dao động theo giai đoạn, nên một cạnh tương quan
trong KG là con số trung bình cả giai đoạn, không phải quy luật bất biến.""" % (
    pct(gr_["2025"], 1, True), pct(gr_["2024"], 1, True), vn(cg["SILVER"], 2),
    vn(cg["USD"], 2), vn(cg["US10Y"], 2)))

w("## 7.2. Knowledge Base")
w("""KB gồm %s tài liệu: %s tin tức từ bảy truy vấn (giá vàng, Fed và lãi suất, ngân hàng
trung ương mua vàng, lạm phát và USD, trú ẩn và địa chính trị, dòng tiền ETF, nhu cầu Trung
Quốc và Ấn Độ), cộng các bản ghi thống kê năm và tương quan sinh từ bảng bằng đúng hai hàm
`stats_to_text`, `corr_text` của chương 6. Từ điển có bốn nhóm: tài sản (GOLD, SILVER, USD,
US10Y, OIL, SP500, VIX, BTC), yếu tố (Fed, lạm phát, ngân hàng trung ương, địa chính trị,
suy thoái, dòng tiền ETF), khu vực (Trung Quốc, Ấn Độ) và chủ đề (kỷ lục, trú ẩn an toàn).""" % (
    num(GD["docs"]), num(GD["news"])))
output("GOLD_LEX = {", squeeze=True)

w("## 7.3. Knowledge Graph: ba nguồn quan hệ")
nbcode("G_gold = new_kg", "G_gold = new_kg", "for a, b, r in DOMAIN")
es = GD["edge_sources"]
w("""Từ tin tức: mỗi tiêu đề có nhiều thực thể thì nối chúng với nhau. Từ bảng: tương quan
tuần giữa 8 tài sản; nếu tin tức đã tạo cạnh giữa hai tài sản thì chỉ gắn thêm thuộc tính
`corr`, không ghi đè trọng số đếm từ tin. Từ tri thức chuyên gia: danh sách `DOMAIN` gồm 9
quan hệ nhân quả đã biết, ví dụ lợi suất trái phiếu là chi phí cơ hội của việc giữ vàng (vàng
không trả lãi); cạnh đã có thì đổi nhãn quan hệ cho rõ nghĩa, chưa có thì thêm với
`source="domain"` (nét đứt trên hình). Kết quả: %s nút, %s cạnh, trong đó %s cạnh từ tin
tức, %s cạnh từ bảng và %s cạnh chuyên gia mới.""" % (
    GD["nodes"], GD["edges"], es.get("text", 0), es.get("table", 0), es.get("domain", 0)))
output("G_gold = new_kg", squeeze=True)
w(fig("figs/kg_gold.png", "KG vàng. Nét liền: tin tức và bảng; nét đứt: tri thức chuyên gia"))

w("## 7.4. Hỏi đáp mẫu và đánh giá")
output("gold_rag = GraphRAG", max_lines=40)
w(fig("figs/eval_gold.png", "Context recall@k, bài vàng", width="55%"))
g4r, g4g = ev("gold", 4, "RAG"), ev("gold", 4, "GraphRAG")
g6r, g6g = ev("gold", 6, "RAG"), ev("gold", 6, "GraphRAG")
check(abs(g4r - g4g) < 1e-9, "vang k=4 ngang nhau")
w("""GOLD bị loại khỏi phần mở rộng vì là hub (có mặt trong hầu hết tài liệu), nên GraphRAG
mở rộng sang các yếu tố như lạm phát, ngân hàng trung ương, địa chính trị. Ở bài này hai chế
độ ngang nhau (recall@4 cùng là %s): hầu hết tài liệu đều có chữ "gold", câu hỏi thường chứa
từ khóa đặc trưng (Fed, ETF, bitcoin) nên truy xuất vector đã đủ tốt. Ở k = 6, RAG thường còn
nhỉnh hơn (%s so với %s), có thể do điểm cộng từ KG đẩy một tài liệu thiếu từ khóa vàng lên
top-k; với 10 câu hỏi, chênh lệch này chỉ là một câu.""" % (vn(g4r), vn(g6r), vn(g6g)))

# ================================================================== 8. DEPLOY
w("# 8. Deploy và demo (mục 4.6)")
w("## 8.1. Đóng gói")
nbcode("DEPLOY = os.path.join", "DEPLOY = os.path.join", "shutil.copy(")
w("""KB, từ điển và KG của mỗi bài được ghi thành một file JSON, kèm bản sao `ragkg_core.py`.
Thư mục `deploy/` vì thế tự chạy được ở bất kỳ đâu có Python (máy cá nhân, server, Hugging
Face Spaces) mà không cần notebook hay dữ liệu gốc.""")

w("## 8.2. Giao diện Gradio")
w(R.py(R.lines("deploy/app.py", "FILES = [", "return (d, q, m, *ask(d, q, m, 4))")))
w("""Mỗi bài được nạp thành một đối tượng `GraphRAG`; `use_embeddings=False` cho app chỉ dùng
TF-IDF để khởi động trong vài giây, không cần tải mô hình. `ask` nhận bài, câu hỏi, chế độ,
top-k từ giao diện và trả về bốn chuỗi: câu trả lời, nguồn, thực thể cùng phần mở rộng KG,
facts từ KG. `from_url` đọc tham số trên đường dẫn (`?domain=gold&q=...&mode=rag`) để mở sẵn
câu hỏi và trả lời ngay khi tải trang; các ảnh demo dưới đây được chụp tự động nhờ cơ chế
này. Phần khai báo giao diện:""")
w(R.py(R.lines("deploy/app.py", "with gr.Blocks", "demo.load(from_url")))
w("""`btn.click(ask, inputs, outputs)` nối nút Hỏi với hàm `ask`: Gradio đọc giá trị bốn ô đầu
vào, gọi hàm, rồi điền bốn giá trị trả về vào bốn ô kết quả theo đúng thứ tự. Trên Kaggle, ô
cuối notebook gọi `demo.launch(share=True)` để có đường dẫn công khai tạm thời; ngoài Kaggle
chạy `python deploy/app.py` rồi mở http://localhost:7860.""")
output("spec = importlib.util", squeeze=False)

w("## 8.3. Demo")
w("""Ba ảnh dưới chụp từ bản đóng gói chạy trên máy cá nhân, chế độ GraphRAG, top-k = 4. Bản
này dùng TF-IDF nên câu trả lời có thể khác chút ít so với notebook (hybrid).""")
w(fig("figs/demo_diabetes.png",
      "Bài tiểu đường: KG mở rộng từ Nephropathy, Cardiovascular disease sang Insulin, "
      "Type 2 diabetes, Hypertension, Obesity", width="92%"))
w(fig("figs/demo_stocks.png",
      "Bài chứng khoán: câu trả lời lấy đúng bản ghi HPG năm 2021 nhờ thực thể năm, kèm facts "
      "về ngành và tương quan", width="92%"))
w(fig("figs/demo_gold.png",
      "Bài vàng: GOLD là hub nên không dùng để mở rộng; facts cho thấy quan hệ Fed -> USD",
      width="92%"))

# ================================================================== 9. SO SANH
w("# 9. So sánh và đánh giá")
w("## 9.1. Tổng hợp kết quả")
eval_fig()
w(fig("figs/eval_summary.png", "Context recall@k của RAG thường và GraphRAG trên ba bài"))
w(table(["Bài", "k", "RAG", "GraphRAG", "Chênh lệch"],
        [[t, k, vn(ev(d, k, "RAG")), vn(ev(d, k, "GraphRAG")),
          ("%+.3f" % (ev(d, k, "GraphRAG") - ev(d, k, "RAG"))).replace(".", ",")]
         for d, t in [("diabetes", "Tiểu đường"), ("stocks", "Chứng khoán"), ("gold", "Vàng")]
         for k in (2, 4, 6)]))

w("## 9.2. Khi nào KG giúp ích")
w("""Kết quả theo một quy luật rõ: GraphRAG có lợi nhất khi **câu hỏi nối nhiều thực thể và
tài liệu liên quan nằm rải rác**, như câu hỏi thuốc - biến chứng ở bài tiểu đường hay công ty -
chủ đề tin ở bài chứng khoán. Khi câu hỏi đã chứa từ khóa rất đặc trưng và gần như mọi tài
liệu đều nói về cùng một chủ thể (bài vàng), truy xuất hybrid đã tìm đủ ngữ cảnh và KG không
thêm được gì. Ngoài điểm số, KG mang lại thứ mà RAG thường không có: các facts có trọng số đi
kèm câu trả lời (thực thể nào liên quan, qua quan hệ gì, mạnh tới đâu), giúp người dùng kiểm
tra và hiểu vì sao hệ thống chọn các tài liệu đó.""")

w("## 9.3. Giới hạn của bài")
w("""* Bộ câu hỏi nhỏ (10-12 câu mỗi bài) do tự soạn: một câu đúng hay sai làm recall đổi 0,08-0,1,
  nên kết quả mang tính minh họa, chưa phải kiểm định thống kê.
* Nhận diện thực thể bằng từ điển chính xác nhưng độ phủ thấp (%s thực thể mỗi đoạn ở bài
  tiểu đường).
* Quan hệ đồng xuất hiện chỉ nói hai thực thể được nhắc cùng nhau, không khẳng định quan hệ
  nhân quả; chỉ bài vàng có quan hệ nhân quả, và đó là tri thức đưa vào bằng tay.
* Tin tức chỉ có tiêu đề; truy vấn PubMed còn lẫn vài bài lạc đề.
* Dữ liệu và câu hỏi bằng tiếng Anh. Kết quả chứng khoán và vàng chỉ minh họa kỹ thuật,
  không phải khuyến nghị đầu tư.""" % vn(DB["ent_per_chunk"], 2))

# ================================================================== 10. KET LUAN
w("# 10. Kết luận")
w("""Bài làm đi qua đủ sáu mục của đề và để lại những kết luận đo được:

1. **KG và KB.** Một KB thống nhất (văn bản, nguồn, thực thể) dựng được cho cả dữ liệu văn
   bản lẫn dữ liệu bảng; bảng phải được chuyển thành câu văn mới đưa vào RAG được, đồng thời
   cung cấp quan hệ định lượng (tương quan) mà văn bản không có.
2. **RAG và GraphRAG.** Truy xuất hybrid (embedding + TF-IDF) là nền tốt; GraphRAG chỉ thêm
   một bước cộng điểm theo thực thể và láng giềng trong KG nhưng tăng context recall@4 từ %s
   lên %s ở bài tiểu đường và từ %s lên %s ở bài chứng khoán, ngang RAG ở bài vàng.
3. **Deploy.** Hệ thống đóng gói thành một thư mục JSON + Python chạy được không cần GPU hay
   API key, có giao diện Gradio và đường dẫn chia sẻ câu hỏi.

Hướng phát triển: dùng LLM hoặc scispaCy để trích thực thể và quan hệ có nhãn (treats,
causes, increases_risk), lưu KG vào Neo4j để truy vấn nhiều bước bằng Cypher, thêm dữ liệu
tiếng Việt với embedding đa ngữ, và đánh giá trên bộ câu hỏi lớn hơn có đo cả chất lượng câu
trả lời do LLM sinh ra.""" % (vn(d4r), vn(d4g), vn(s4r), vn(s4g)))

# ================================================================== 11. CHAY LAI
w("# 11. Cách chạy lại toàn bộ bài")
w("""**Trên Kaggle:** Create, New Notebook, File, Import Notebook, chọn `RAG_KG_Kaggle.ipynb`;
bật Settings, Internet, On; Run All. Notebook tự cài thư viện và tải dữ liệu mới; muốn dùng
đúng bản chụp trong `data/` thì upload thư mục này thành Kaggle Dataset và gắn vào notebook.

**Trên máy cá nhân** (chạy lại notebook với dữ liệu đã chụp, rồi sinh lại báo cáo):""")
w("""```
set LOCAL_DATA=data
set WORK_DIR=.
jupyter nbconvert --to notebook --execute --inplace RAG_KG_Kaggle.ipynb
python tools/build_report.py
python deploy/app.py
```""")
w("""Lệnh `build_report.py` cần pandoc để xuất `.docx` và Microsoft Word để cập nhật mục lục,
xuất `.pdf`.""")

w("# Tài liệu tham khảo")
w("""1. Lý Hải Quân (2026). *KG_RAG: Knowledge Graph và RAG cho bệnh tiểu đường, chứng khoán,
   vàng* (mã nguồn, notebook, dữ liệu, demo). """ + REPO + """
2. Lewis P. và cộng sự (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP
   Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
3. Edge D. và cộng sự (2024). *From Local to Global: A Graph RAG Approach to Query-Focused
   Summarization*. Microsoft Research. https://arxiv.org/abs/2404.16130
4. Hogan A. và cộng sự (2021). *Knowledge Graphs*. ACM Computing Surveys 54(4).
   https://arxiv.org/abs/2003.02320
5. Singhal A. (2012). *Introducing the Knowledge Graph: things, not strings*. Google Blog.
6. NCBI E-utilities (PubMed), yfinance, Google News RSS, sentence-transformers
   (all-MiniLM-L6-v2), NetworkX, scikit-learn, Gradio.""")


# ------------------------------------------------------------------ xuat file
def main():
    md = "\n".join(MD).rstrip() + "\n"
    OUT_MD.write_text(md, encoding="utf-8")
    print("%s: %d ky tu, %d khoi code, %d hinh" % (OUT_MD.name, len(md), md.count("```python"),
                                                  FIG_NO[0]))
    ref = HERE / "ref" / "reference_code.docx"
    cmd = ["pandoc", str(OUT_MD), "-o", str(OUT_DOCX), "--toc", "--toc-depth=2",
           "--syntax-highlighting=tango", "--resource-path", str(ROOT),
           "--reference-doc", str(ref)]
    subprocess.run(cmd, check=True)
    print("%s: %.0f KB" % (OUT_DOCX.name, OUT_DOCX.stat().st_size / 1024))
    if "--no-pdf" in sys.argv:
        return
    ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
          str(HERE / "export_pdf.ps1"), "-DocPath", str(OUT_DOCX), "-PdfPath", str(OUT_PDF)]
    r = subprocess.run(ps, capture_output=True, text=True, timeout=600)
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
