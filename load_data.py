def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        label = []
        data = []
        line = f.readline()
        dic = {'negative': 0, 'positive': 1}

        for line in f:
            line = line.replace(' ', '')
            line = line.strip().split(',')
            if line[0].startswith('@'):
                continue
            label.append(dic.get(line[-1]))
            line_data = [float(x) for x in line[:-2]]
            data.append(line_data)

        return data, label
