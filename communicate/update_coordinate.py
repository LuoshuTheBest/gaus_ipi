def write_coordinate_to_file(target_file, coordinate_info):
    with open(target_file, "w") as tf:
        for i in range(len(coordinate_info)):
            for j in range(len(coordinate_info[i])):
                tf.write(coordinate_info[i][j])
                tf.write("\t")
            tf.write("\n")


def read_coordinate(coordinate_file):
    coordinates = []
    with open(coordinate_file, "r") as cf:
        lines = cf.readlines()
        for line in lines:
            if len(line) <= 1:
                continue
            if "O2H" in line:
                continue
            if line[0] not in "OH":
                continue
            values = line.split()
            values.pop(0)
            if len(values) != 3:
                raise ValueError("Invalid number of dimensions.")
            coordinates.append(values)
    return coordinates


if __name__ == "__main__":
    print("The coordinate files have been read.")
    try:
        coordinate_from_file = read_coordinate("/home/luoshu/PycharmProjects/gaus_ipi/1.com")
        target_file_path = "/home/luoshu/PycharmProjects/gaus_ipi/communicate/coordinate.txt"
        write_coordinate_to_file(target_file_path, coordinate_from_file)
    except ValueError:
        warning = "The number of coordinate mismatch with the presuppose value. "
        warning += "Update the update_coordinate.py file to solve the problem.\n"
        warning += "The following result should be meaningless."
        print(warning)
