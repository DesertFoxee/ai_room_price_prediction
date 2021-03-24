import pandas as pd
import re
import common.config as cf
import common.utils  as utils



folder_out = "/"

def preprocessing_data_nhachoto(path_file):
    df = pd.read_csv(path_file);


def preprocessing_data_phongtro123(path_file):
    df = pd.read_csv(path_file)

    # #Trường thời gian :  loại bỏ các ký tự không phải số và /
    # df[cf.data_fields_scraping[0]] = df[cf.data_fields_scraping[0]].map(lambda x: re.sub('[^0-9//]', '', x))
    #
    # #Trường giá phòng : loại bỏ các ký tự không phải số và .
    # df[cf.data_fields_scraping[1]] = df[cf.data_fields_scraping[1]].map(lambda x: re.sub('[^0-9.]', '', x))
    #
    # #Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    # df[cf.data_fields_scraping[2]] = df[cf.data_fields_scraping[2]].map(lambda x: x[:-2]) # m2
    # df[cf.data_fields_scraping[2]] = df[cf.data_fields_scraping[2]].map(lambda x: re.sub('[^0-9.]', '', x))

    file_name = utils.get_file_name_from_path(path_file)
    file_name = file_name+'_pre.csv'
    path_pre = folder_out+file_name
    print(file_name)
    return


def main():
    # load csv
    preprocessing_data_phongtro123("../urls_phongtro123_data_23032021.csv")


# Hàm main
if __name__ == "__main__":
    main()