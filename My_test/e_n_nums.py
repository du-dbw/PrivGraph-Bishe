def count_graph(file_path):
    nodes = set()
    edge_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            u, v = parts[0], parts[1]

            nodes.add(u)
            nodes.add(v)
            edge_count += 1

    print("节点数:", len(nodes))
    print("边数:", edge_count)


# 示例调用
if __name__ == "__main__":
    file_path = "data\CA-HepPh.txt"
    count_graph(file_path)