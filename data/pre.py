#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

def load_nodes(node_path: Path) -> set[str]:
    """
    Đọc file danh sách đỉnh:
    - Mỗi dòng có thể chứa 1 hoặc nhiều đỉnh
    - Phân tách bởi whitespace (space/tab)
    - Bỏ qua dòng trống và dòng bắt đầu bằng '#'
    Trả về tập node_id (string).
    """
    nodes: set[str] = set()
    with node_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # split() tách theo mọi whitespace: space, tab, nhiều khoảng trắng...
            for tok in s.split():
                if tok.startswith("#"):
                    break  # phần còn lại là comment cùng dòng (nếu có)
                nodes.add(tok)
    return nodes

def filter_edges(node_set: set[str], edge_path: Path, out_path: Path) -> tuple[int, int]:
    """
    Đọc file danh sách cạnh: mỗi dòng "u v" (có thể kèm thêm cột).
    Nếu u hoặc v không thuộc node_set => bỏ qua.
    Ghi ra out_path các dòng hợp lệ (giữ nguyên phần còn lại của dòng).
    """
    kept = 0
    dropped = 0

    with edge_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            s = raw.strip()

            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 2:
                dropped += 1
                continue

            u, v = parts[0], parts[1]
            if (u in node_set) and (v in node_set):
                fout.write(raw + "\n")
                kept += 1
            else:
                dropped += 1

    return kept, dropped

def main():
    parser = argparse.ArgumentParser(description="Filter edge list by a given node list.")
    parser.add_argument("--nodes", required=True, help="Path to node list file (tokens separated by space/tab).")
    parser.add_argument("--edges", required=True, help="Path to edge list file (each line: u v [..]).")
    parser.add_argument("--out", required=True, help="Path to output filtered edge list file.")
    args = parser.parse_args()

    node_set = load_nodes(Path(args.nodes))
    kept, dropped = filter_edges(node_set, Path(args.edges), Path(args.out))

    print(f"Loaded nodes: {len(node_set)}")
    print(f"Edges kept: {kept}")
    print(f"Edges dropped: {dropped}")
    print(f"Output written to: {args.out}")

if __name__ == "__main__":
    main()
