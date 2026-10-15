"""Tien ich trich code THAT tu file nguon de nhung vao bao cao.

Nho vay code trong bao cao luon dong bo voi code trong notebook:
khong co chuyen "bao cao viet mot dang, notebook chay mot dang khac".
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def obj(rel, *names, dedent=True):
    """Trich source cua class/def theo ten (dung AST nen chinh xac tuyet doi)."""
    src = _read(rel)
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for name in names:
        found = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                found = node
                break
        if found is None:
            raise KeyError(f"{name} khong co trong {rel}")
        start = min([d.lineno for d in found.decorator_list] + [found.lineno]) - 1
        out.append("\n".join(lines[start:found.end_lineno]))
    return "\n\n\n".join(out)


def cell(rel, needle, drop_comment_header=False):
    """Trich mot code-cell (dinh dang percent) chua chuoi `needle`."""
    src = _read(rel)
    parts = re.split(r"(?m)^# %%.*$", src)
    heads = re.findall(r"(?m)^# %%.*$", src)
    for head, body in zip(heads, parts[1:]):
        if "[markdown]" in head:
            continue
        if needle in body:
            text = body.strip("\n")
            if drop_comment_header:
                text = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
            return text.strip("\n")
    raise KeyError(f"Khong tim thay cell chua {needle!r} trong {rel}")


def lines(rel, first, last):
    """Trich doan code theo hai chuoi moc (bao gom ca hai dong moc)."""
    src = _read(rel).splitlines()
    i = next(i for i, l in enumerate(src) if first in l)
    j = next(j for j in range(i, len(src)) if last in src[j])
    return "\n".join(src[i:j + 1])


def nbout(rel_nb, needle, max_lines=None):
    """Lay phan text in ra cua cell (trong notebook DA CHAY) chua chuoi `needle`."""
    import json
    nb = json.loads(_read(rel_nb))
    for c in nb["cells"]:
        src = "".join(c.get("source", []) if isinstance(c.get("source"), list)
                      else c.get("source", ""))
        if c["cell_type"] != "code" or needle not in src:
            continue
        texts = []
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream" and o.get("name") != "stderr":
                texts.append("".join(o.get("text", [])))
            elif o.get("output_type") in ("execute_result", "display_data"):
                d = o.get("data", {})
                plain = "".join(d.get("text/plain", ""))
                # bo dong mo ta hinh cua matplotlib, vi du "<Figure size 1200x360 with 2 Axes>"
                if plain and not plain.startswith("<Figure size"):
                    texts.append(plain.rstrip("\n") + "\n")
        text = "".join(texts).rstrip("\n")
        if max_lines:
            ls = text.splitlines()
            if len(ls) > max_lines:
                text = "\n".join(ls[:max_lines] + ["..."])
        return text
    raise KeyError(f"Khong tim thay cell chua {needle!r} trong {rel_nb}")


def nbcell(rel_nb, needle, first=None, last=None):
    """Trich source cua code-cell trong notebook chua chuoi `needle`.

    `first`/`last`: chi lay doan tu dong chua `first` toi dong chua `last`."""
    import json
    nb = json.loads(_read(rel_nb))
    for c in nb["cells"]:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if c["cell_type"] != "code" or needle not in src:
            continue
        ls = [l for l in src.splitlines() if not l.startswith("%%writefile")]
        if first:
            i = next(i for i, l in enumerate(ls) if first in l)
            j = next(j for j in range(i, len(ls)) if last in ls[j]) if last else len(ls) - 1
            ls = ls[i:j + 1]
        return "\n".join(ls).strip("\n")
    raise KeyError(f"Khong tim thay cell chua {needle!r} trong {rel_nb}")


def nbobj(rel_nb, needle, *names):
    """Trich def/class theo ten tu code-cell chua `needle` (dung AST nhu `obj`)."""
    src = nbcell(rel_nb, needle)
    tree = ast.parse(src)
    lines = src.splitlines()
    out = []
    for name in names:
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        out.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    return "\n\n\n".join(out)


def nbstmt(rel_nb, needle, first, last):
    """Trich cac cau lenh top-level tu cau lenh chua `first` toi cau lenh chua `last`."""
    src = nbcell(rel_nb, needle)
    tree = ast.parse(src)
    lines = src.splitlines()
    stmts = tree.body
    seg = lambda n: "\n".join(lines[n.lineno - 1:n.end_lineno])
    i = next(k for k, n in enumerate(stmts) if first in seg(n))
    j = next(k for k in range(i, len(stmts)) if last in seg(stmts[k]))
    return "\n".join(lines[stmts[i].lineno - 1:stmts[j].end_lineno])


def py(code):
    return "```python\n" + code.rstrip() + "\n```"


def out(text, width=88):
    """Boc output cua notebook; ngat cac dong qua dai de khong tran le trong DOCX."""
    import textwrap
    wrapped = []
    for line in text.strip("\n").splitlines():
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width,
                                         subsequent_indent="    ",
                                         break_long_words=True,
                                         break_on_hyphens=False) or [""])
    return "```\n" + "\n".join(wrapped) + "\n```"
