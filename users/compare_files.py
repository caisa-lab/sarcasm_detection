def read_file(file):
    with open(file, 'r') as f:
       ids = [id for id in f.readlines()]
    
    return ids

spirs_id = set(read_file('users.txt'))
kims_id = set(read_file('kim_users.txt'))

print(len(spirs_id), len(kims_id))
print(len(spirs_id.intersection(kims_id)))
