

files = ["4dfalse.txt",
        "4dTrue2.txt",
        "rootfalse.txt",
        "roottrue.txt"]

my_dict = {}

for batch_file in files:
    cur_file = open(batch_file)
    content = cur_file.readline()
    content = content.split("['")[1]
    content = content.split("']")[0]
    content = content.split("\', \'")
    for vid in content:
        repos = vid.split('/')
        elem = repos[len(repos)-1].split('.')[0]
        if elem not in my_dict:
            my_dict[elem] = 1
        else:
            my_dict[elem] = my_dict[elem] + 1
    cur_file.close()

print(my_dict)