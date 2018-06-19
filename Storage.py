import numpy as np
import matplotlib.pyplot as plt
import struct
import os


def write_params(path, dic):
    f = open(path, 'wb')
    f.write(struct.pack('I', len(dic)))
    for x in dic.keys():
        f.write(struct.pack('III', len(x), len(dic[x].shape), dic[x].nbytes))
    for x in dic.keys():
        f.write(x.encode())
        for d in dic[x].shape:
            f.write(struct.pack('I', d))
        f.write(dic[x].tobytes())
    f.close()


def read_params(path):
    dic = dict()
    f = open(path, 'rb')
    count = struct.unpack('I', f.read(4))[0]
    names = []
    for i in range(0, count):
        name, shape, length = struct.unpack('III', f.read(12))
        names.append([name, shape, length])

    for x in names:
        name = f.read(x[0]).decode()
        shape = []
        for i in range(0, x[1]):
            shape.append(struct.unpack('I', f.read(4))[0])
        data = f.read(x[2])
        arr = np.frombuffer(data, dtype=np.float32)
        arr = arr.reshape(shape)
        dic[name] = arr
    f.close()

    return dic


def enum_samples(path_list):
    file_list = dict()

    for path in path_list:
        if path[1] not in file_list.keys():
            file_list[path[1]] = list()
        for file in os.listdir(path[0]):
            file_list[path[1]].append(path[0] + '/' + file)
    return file_list


def pick_some(dic, count):
    sample_list = list()
    label_list = list()

    kind_list = np.random.randint(0, len(dic.keys()), count)

    for item in kind_list:
        img_idx = np.random.randint(0, len(dic[item]), 1)[0]
        img = plt.imread((dic[item][img_idx]))
        lbl = np.zeros(len(dic.keys()), np.float32)
        lbl[item] = 1
        sample_list.append(img)
        label_list.append(lbl)

    return sample_list, label_list


def pick_one_indexed(dic, index):
    remain = index
    kind = 0
    while remain >= len(dic[kind]):
        remain -= len(dic[kind])
        kind = kind + 1
    path = dic[kind][remain]
    img = plt.imread(path)
    lbl = np.zeros(len(dic.keys()), np.float32)
    lbl[kind] = 1
    return [img], [lbl]


if __name__ == '__main__':
    path_list = [['./TestSamples/Ges_0', 0],
                 ['./TestSamples/Ges_1', 1],
                 ['./TestSamples/Ges_2', 2],
                 ['./TestSamples/Ges_3', 3],
                 ['./TestSamples/Ges_3-A', 3],
                 ['./TestSamples/Ges_3-B', 3],
                 ['./TestSamples/Ges_4', 4],
                 ['./TestSamples/Ges_5', 5]
                 ]
    file_list = enum_samples(path_list)
    # sp, lb = pick_some(file_list, 10)

    for x in range(0, 1000):
        print(x)
        i, l = pick_one_indexed(file_list, x)
        print(i, l)


# def read_from_db(path):
#     db = open(path, 'rb')
#     width, height, kind_num = struct.unpack('III', db.read(12))
#     kind1, kind2, kind3 = struct.unpack('III', db.read(12))
#     total = kind1 + kind2 + kind3
#
#     data = bytearray(db.read())
#     data = np.array(data)
#     data = data.reshape([total, width, height, 1])
#
#     # for x in range(0, 3):
#     #     plt.imshow(data[x], cmap='gray')
#     #     plt.show()
#
#     return [data, [kind1, kind2, kind3]]
#
#
# def pick_some(db, count):
#     total = 0
#     indices = []
#     for x in range(0, len(db[1])):
#         total = total + db[1][x]
#         indices.append(total)
#
#     nums = np.random.randint(0, total, count)
#     data = []
#     labels = []
#
#     # print(nums)
#
#     for x in range(0, len(nums)):
#         index = nums[x]
#         data.append(db[0][index])
#         label = [0, 0, 0]
#         if index < indices[0]:
#             label[0] = 1
#         elif index < indices[1]:
#             label[1] = 1
#         else:
#             label[2] = 1
#         labels.append(label)
#
#     return data, labels
#
#
# if __name__ == '__main__':
#     db = read_from_db('./LinkDB/mono.db')
#     data, labels = pick_some(db, 10)
#     print(data)
#     print(labels)
