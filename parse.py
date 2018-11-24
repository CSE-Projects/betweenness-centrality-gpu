with open("graph1") as f:
    with open("graph1out", "w") as f1:
        i = 0
        for line in f:
            if i == 0: 
                f1.write(line)
            else:
                line = [x - 1 for x in map(int, line.split())]
                # for item in line:
                f1.write(' '.join(str(x) for x in line))
                f1.write('\n')
            i += 1
