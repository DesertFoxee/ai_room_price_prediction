import pandas as pd
import re
import common.config as cf


def preprocessing_data_nhachoto(path_file):
    df = pd.read_csv(path_file);


def preprocessing_data_phongtro123(path_file):
    df = pd.read_csv(path_file)

    #Trường thời gian :  loại bỏ các ký tự không phải số và /
    df[cf.data_fields_scraping[0]] = df[cf.data_fields_scraping[0]].map(lambda x: re.sub('[^0-9//]', '', x))

    #Trường giá phòng : loại bỏ các ký tự không phải số và .
    df[cf.data_fields_scraping[1]] = df[cf.data_fields_scraping[1]].map(lambda x: re.sub('[^0-9.]', '', x))

    #Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    df[cf.data_fields_scraping[2]] = df[cf.data_fields_scraping[2]].map(lambda x: x[:-2]) # m2
    df[cf.data_fields_scraping[2]] = df[cf.data_fields_scraping[2]].map(lambda x: re.sub('[^0-9.]', '', x))


    print(df[cf.data_fields_scraping[2]])
    return


def main():
    # load csv
    preprocessing_data_phongtro123("../urls_phongtro123_data_23032021.csv")


# Hàm main
if __name__ == "__main__":
    main()