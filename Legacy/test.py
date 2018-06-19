import os

logs_dir = './logs_pb/'
test_image_path = './SAMPLE/Paper/'
halfSize = 96
fullSize = 192
batch_size = 16


def get_lastest_pbfile(path):
    lastest_pbfile = ""
    for target in os.listdir(path):
        file = str(path + target)
        if os.path.isfile(file):
            create_time = 0
            if (os.path.splitext(file)[-1] == '.pb') and (create_time < os.stat(file).st_ctime):
                create_time = os.stat(file).st_ctime
                lastest_pbfile = file
    return lastest_pbfile


def get_images_and_label(path, required_number, required_format):
    num = 0
    image_batch = []
    label_batch = []
    for target in os.listdir(path):
        if num < required_number:
            file = str(path + target)
            if os.path.splitext(file)[-1] == required_format:
                image_batch.append(file)
            else:
                print("error")
                quit()
            if 'Rock' in path:
                label_batch.append(0)
            if 'Scissors' in path:
                label_batch.append(1)
            if 'Paper' in path:
                label_batch.append(2)
            num += 1
    #         print("1")
    # print("2")
    return image_batch, label_batch


if __name__ == '__main__':
    image, label = get_images_and_label(test_image_path, 16, '.png')
    print(image, label)