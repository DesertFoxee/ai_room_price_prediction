import pandas as pd
import re
import common.config as cf
import common.utils as utils
from datetime import datetime, timedelta

data_convert_time = [
        ["phút", 0  ],
        ["giờ" , 0  ],
        ["hôm" , 0  ],

        ["ngày", 1  ],
        ["tuần", 7  ],
        ["tháng",30 ],
        ["năm" , 365]
    ]

folder_out = "../data_pre/"
folder_out_spider= "../data_pre/data_daily/"

path_data_raw = 'data_pre/data_train/data_phongtro123_data.csv'
path_data_out = 'housepricedata.csv'


def drop_row(data_frame, start_index , end_index):
    data_frame.drop(data_frame.index[0: start_index], inplace=True)
    data_frame.drop(data_frame.index[end_index: len(data_frame)], inplace=True)


#lấy thời gian đơn vị
def get_pre_month_year_from_str(date_get, str_time):
    match = [arr_cv_time for arr_cv_time in data_convert_time if arr_cv_time[0] in str_time]
    if match:
        factor = re.sub("\\D", "", str_time)
        if not factor:
            factor = "0"
        day_sub = int(factor) * match[0][1]
        pre_time = date_get - timedelta(days=day_sub)
        str_date = pre_time.strftime("%m/%Y")
        return str_date
    return ""


def save_file(df, path_file_out):
    file_name = utils.get_file_name_from_path(path_file_out)
    file_name = file_name+'_pre.csv'
    path_pre = folder_out+file_name
    df.to_csv(path_pre, mode='w', header=True, index=False)


def preprocessing_data_nhachoto(path_file ,istart, iend):
    df = pd.read_csv(path_file)
    # drop_row(df , istart, iend) # chỉ lấy từ dòng istart->iend
    file_name = utils.get_file_name_from_path(path_file)

    col_time = cf.field_header_file_data[1]
    col_price = cf.field_header_file_data[2]
    col_acreage = cf.field_header_file_data[3]
    col_address = cf.field_header_file_data[4]
    col_type = cf.field_header_file_data[5]
    col_detail = cf.field_header_file_data[6]

    str_data_get = file_name[file_name.rindex("_")+1:]
    data_get = datetime.strptime(str_data_get, "%d%m%Y")
    df[col_time] = df[col_time].map(lambda x: get_pre_month_year_from_str(data_get, x))
    df.drop(columns=col_detail, inplace=True)

    # Trường giá phòng : loại bỏ các ký tự không phải số và . + giá lớn hơn 0
    df[col_price] = df[col_price].map(lambda x: x[0: x.rindex("-")])
    df[col_price] = df[col_price].map(lambda x: x.replace("triệu/tháng", "").replace(",", "."))
    df[col_price]= df[col_price].map(lambda x: re.sub('[^0-9.]', '', x))
    df[col_price]= df[col_price].map(lambda x: float(x)*1000000 if float(x) < 100 else float(x)*1000)
    df[col_price] = df[col_price].map(lambda x: int(x))
    df = df[df[col_price] > 0]

    # Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    df[col_acreage] = df[col_acreage].map(lambda x: x[:-2])  # m2
    df[col_acreage] = df[col_acreage].map(lambda x: re.sub('[^0-9.]', '', x))

    file_name = file_name + '_pre.csv'
    path_pre = folder_out + file_name
    df.to_csv(path_pre, mode='w', header=True, index=False)
    return

def preprocessing_data_phongtro123(path_file):
    df = pd.read_csv(path_file)
    # drop_row(df , istart, iend) # chỉ lấy từ dòng istart->iend

    col_time = cf.field_header_file_data[1]
    col_price = cf.field_header_file_data[2]
    col_acreage = cf.field_header_file_data[3]
    col_address = cf.field_header_file_data[6]
    col_detail = cf.field_header_file_data[7]

    #Trường thời gian :  loại col_timebỏ các ký tự không phải số và /
    df[col_time] = df[col_time].map(lambda x: re.sub('[^0-9//]', '', x))
    df[col_time] = df[col_time].map(lambda x: x[3:])

    #Trường giá phòng : loại bỏ các ký tự không phải số và . + giá lớn hơn 0
    df = df[df[col_price].str.contains("Thỏa thuận") == False]
    df[col_price]= df[col_price].map(lambda x: re.sub('[^0-9.]', '', x))
    df[col_price]= df[col_price].map(lambda x: float(x)*1000000 if float(x) < 100 else float(x)*1000)
    df[col_price] = df[col_price].map(lambda x: int(x))
    df = df[df[col_price] > 0]

    #Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    df[col_acreage] = df[col_acreage].map(lambda x: x[:-2]) # m2
    df[col_acreage] = df[col_acreage].map(lambda x: re.sub('[^0-9.]', '', x))

    # Trường địa chỉ : bỏ từ Địa chỉ: trong dữ liệu
    df[col_address] = df[col_address].map(lambda x: x.replace("Địa chỉ: ", ""))

    # Trường chi tiết : bỏ từ các từ không cần thiết
    df[col_detail] = df[col_detail].str.lower()
    df[col_detail] = df[col_detail].map(lambda x: x.replace(" ", "").replace("‎", "").replace("‬", "").replace("‪", ""))
    df[col_detail] = df[col_detail].map(lambda x: x.replace("mình", "").replace("phòng trọ", "").replace("cho thuê", "").replace("nhà trọ", ""))
    df[col_detail] = df[col_detail].map(lambda x: x.replace("phòng", "").replace("vị trí", "").replace("cho thuê", "").replace("diện tích", ""))

    #Trường có khép kín hay không:
    # df[col_type] = df[col_price]<= 1000000

    file_name = utils.get_file_name_from_path(path_file)
    file_name = file_name+'_pre.csv'
    path_pre = folder_out+file_name

    df = df[cf.field_header_file_data]
    df.to_csv(path_pre, mode='w', header=True, index=False)
    return


def preprocessing_data_phongtro123_spider(path_file):
    df = pd.read_csv(path_file)
    # drop_row(df , istart, iend) # chỉ lấy từ dòng istart->iend

    # col_time = cf.field_header_file_spider[1]
    col_price = cf.field_header_file_spider[3]
    col_acreage = cf.field_header_file_spider[4]
    col_address = cf.field_header_file_spider[8]
    col_detail = cf.field_header_file_spider[9]

    #Trường thời gian :  loại col_timebỏ các ký tự không phải số và /
    # df[col_time] = df[col_time].map(lambda x: re.sub('[^0-9//]', '', x))
    # df[col_time] = df[col_time].map(lambda x: x[3:])

    #Trường giá phòng : loại bỏ các ký tự không phải số và . + giá lớn hơn 0
    df = df[df[col_price].str.contains("Thỏa thuận") == False]
    df[col_price]= df[col_price].map(lambda x: re.sub('[^0-9.]', '', x))
    df[col_price]= df[col_price].map(lambda x: float(x)*1000000 if float(x) < 100 else float(x)*1000)
    df[col_price] = df[col_price].map(lambda x: int(x))
    df = df[df[col_price] > 0]

    #Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    df[col_acreage] = df[col_acreage].map(lambda x: x[:-2]) # m2
    df[col_acreage] = df[col_acreage].map(lambda x: re.sub('[^0-9.]', '', x))

    # Trường địa chỉ : bỏ từ Địa chỉ: trong dữ liệu
    df[col_address] = df[col_address].map(lambda x: x.replace("Địa chỉ: ", ""))

    # Trường chi tiết : bỏ từ các từ không cần thiết
    df[col_detail] = df[col_detail].str.lower()
    df[col_detail] = df[col_detail].map(lambda x: x.replace(" ", "").replace("‎", "").replace("‬", "").replace("‪", ""))

    #Trường có khép kín hay không:
    # df[col_type] = df[col_price]<= 1000000

    file_name = utils.get_file_name_from_path(path_file)
    file_name = file_name+'_pre.csv'
    path_pre = folder_out_spider+file_name

    df = df[cf.field_header_file_data]
    df.to_csv(path_pre, mode='w', header=True, index=False)
    return


# Xử lý các cột trong dữ liệu thô sang file housepricedata_1.csv
def convert_data_row():
    df = pd.read_csv(path_data_raw)
    df.drop(['stt', 'chitiet', 'diachi'], axis=1, inplace=True)
    df["thoigian"] = pd.to_datetime(df["thoigian"])
    df.insert(1, 'thang', df["thoigian"].dt.month)
    df.insert(0, 'nam', df["thoigian"].dt.year)
    df = df.drop(['thoigian'], axis=1)
    df.to_csv(path_data_out, index=False,header=True)


def main():
    # load csv
    # preprocessing_data_phongtro123("../data_bk/data_phongtro123_27032021.csv")

    # preprocessing_data_nhachoto("../data_bk/data_nhachoto_26032021.csv",0,3257)


    convert_data_row()
    return


# Hàm main
if __name__ == "__main__":
    main()