import os


def get_lastest_pbfile(path):
    lastest_pbfile = ""
    for target in os.listdir(path):
        file = str(str(path + target))
        if os.path.isfile(file):
            create_time = 0
            if (os.path.splitext(file)[-1] == '.pb') and (create_time < os.stat(file).st_ctime):
                create_time = os.stat(file).st_ctime
                lastest_pbfile = file
    return lastest_pbfile


if __name__ == '__main__':
    # print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
    logs_dir = './logs_pb/'

    print(get_lastest_pbfile(logs_dir))
