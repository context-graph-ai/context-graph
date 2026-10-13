"""Scan public Rust items and their local dependencies for named placeholder SQL."""
import json
import re
import sys
from pathlib import Path


def tokens(text):
    """Discard comments; preserve strings as single tokens with their source lines."""
    result = []
    i = 0
    while i < len(text):
        start = i
        if text[i].isspace():
            i += 1
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        if text.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(text) and depth:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue
        raw = re.match(r'(?:br|r)(#*)"', text[i:])
        if raw:
            begin = i + raw.end()
            terminator = '"' + raw[1]
            end = text.find(terminator, begin)
            if end < 0:
                raise ValueError("unterminated raw string")
            value = text[begin:end]
            i = end + len(terminator)
            kind = "string"
        elif text[i] == '"' or text.startswith('b"', i):
            i += 2 if text.startswith('b"', i) else 1
            begin = i
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    break
                else:
                    i += 1
            value = text[begin:i].replace("\\n", " ").replace("\\\"", '"')
            i += 1
            kind = "string"
        else:
            # Character literals are not lifetimes and may contain brace characters.
            char = re.match(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\])'", text[i:])
            word = re.match(r"[A-Za-z_][A-Za-z_0-9]*", text[i:])
            if char:
                value = char[0]
                i += len(value)
                kind = "char"
            elif word:
                value = word[0]
                i += len(value)
                kind = "word"
            else:
                value = text[i]
                i += 1
                kind = "punctuation"
        result.append((kind, value, text.count("\n", 0, start) + 1))
    return result


def closing(ts, start, left="{", right="}"):
    depth = 0
    for i in range(start, len(ts)):
        if ts[i][0] != "punctuation":
            continue
        if ts[i][1] == left:
            depth += 1
        elif ts[i][1] == right:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unclosed Rust delimiter")


def non_test_tokens(ts):
    """Exclude cfg(test) modules, preserving ordinary source and feature-gated APIs."""
    result = []
    i = 0
    while i < len(ts):
        if ts[i][1] == "#" and i + 1 < len(ts) and ts[i + 1][1] == "[":
            end = closing(ts, i + 1, "[", "]")
            attr = "".join(t[1] for t in ts[i + 2:end])
            if attr == "cfg(test)":
                j = end + 1
                while j < len(ts) and ts[j][1] not in ("{", ";"):
                    j += 1
                if j < len(ts) and ts[j][1] == "{":
                    i = closing(ts, j) + 1
                    continue
                if j < len(ts):
                    i = j + 1
                    continue
        result.append(ts[i])
        i += 1
    return result


def items(path, root):
    ts = non_test_tokens(tokens(path.read_text()))
    found = []
    for i, token in enumerate(ts):
        if token[0] != "word" or token[1] not in ("fn", "const", "static"):
            continue
        if i + 1 >= len(ts) or ts[i + 1][0] != "word":
            continue
        name = ts[i + 1][1]
        j = i + 2
        while j < len(ts) and ts[j][1] not in ("{", ";", "="):
            j += 1
        if j == len(ts) or ts[j][1] == ";":
            continue
        if token[1] == "fn":
            if ts[j][1] != "{":
                continue
            end = closing(ts, j)
        else:
            if ts[j][1] != "=":
                continue
            end = j + 1
            while end < len(ts) and ts[end][1] != ";":
                if ts[end][1] == "{":
                    end = closing(ts, end)
                end += 1
        k = i - 1
        while k >= 0 and ts[k][1] in ("async", "unsafe", "extern"):
            k -= 1
        public = k >= 0 and ts[k][1] == "pub"
        body = ts[j + 1:end]
        found.append({"name": name, "path": str(path.relative_to(root)),
                      "line": token[2], "public": public, "body": body})
    return found


def scan(root):
    definitions = []
    for path in sorted((root / "crates/context-graph/src").rglob("*.rs")):
        definitions.extend(items(path, root))
    by_name = {}
    for index, item in enumerate(definitions):
        by_name.setdefault(item["name"], []).append(index)
    violations = []
    pattern = re.compile(r'\b(?:FROM|JOIN|INTO|UPDATE|USING)\s+["`\[]?(signatures|blueprints|approvals)\b', re.I)
    for start, entry in enumerate(definitions):
        if not entry["public"]:
            continue
        todo = [(start, [entry["name"]])]
        seen = set()
        while todo:
            index, chain = todo.pop()
            if index in seen:
                continue
            seen.add(index)
            item = definitions[index]
            for kind, value, line in item["body"]:
                if kind == "string":
                    for match in pattern.finditer(value):
                        violations.append({"api": entry["name"], "api_path": entry["path"],
                                           "path": item["path"], "line": line,
                                           "table": match[1].lower(), "chain": chain})
                elif kind == "word":
                    for callee in by_name.get(value, []):
                        # Functions/constants in the same module or store modules are reachable.
                        destination = definitions[callee]
                        if destination["path"] == item["path"] or "/src/store/" in destination["path"]:
                            todo.append((callee, chain + [value]))
    return sorted(violations, key=lambda v: (v["api_path"], v["api"], v["path"], v["line"], v["table"]))


if __name__ == "__main__":
    print(json.dumps(scan(Path(sys.argv[1]))))
