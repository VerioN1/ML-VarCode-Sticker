def get_labels():
    with open('converted_keras/labels.txt') as f:
        lines = f.readlines()
        arr = []
        for line in lines:
            new_line = line.replace("0 ", "")
            new_line = new_line.replace("1 ", "")
            new_line = new_line.replace("\n", "")
            arr.append(new_line)
        print(arr)
        return arr



