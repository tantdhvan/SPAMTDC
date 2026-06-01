# renumber_edges.py

input_file = "gplus_combined.txt"
output_file = "gplus_renumbered.txt"
mapping_file = "gplus_mapping.txt"

id_map = {}
next_id = 0

def get_new_id(old_id):
    global next_id
    if old_id not in id_map:
        id_map[old_id] = next_id
        next_id += 1
    return id_map[old_id]

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        u, v = line.split()
        new_u = get_new_id(u)
        new_v = get_new_id(v)

        fout.write(f"{new_u} {new_v}\n")

with open(mapping_file, "w") as fmap:
    for old_id, new_id in id_map.items():
        fmap.write(f"{old_id} {new_id}\n")

print("Số đỉnh:", len(id_map))
print("File cạnh mới:", output_file)
print("File ánh xạ:", mapping_file)