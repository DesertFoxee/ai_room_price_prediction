import pandas as pd
import re
import common.config as cf
import common.utils as utl
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


folder_out_spider= "data_pre/data_daily/"

path_data_raw = 'data_train/data_phongtro123_data.csv'
path_data_train = 'roomdata.csv'

path_data_raw2 = 'data_train/data_phongtro123_databk.csv'
path_data_train2 = 'roomdata2.csv'
path_data_imp = 'roomdata2_imp.csv'



def drop_row(data_frame, start_index , end_index):
    data_frame.drop(data_frame.index[0: start_index], inplace=True)
    data_frame.drop(data_frame.index[end_index: len(data_frame)], inplace=True)


# Lấy thời gian đơn vị
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
    file_name = utl.get_file_name_from_path(path_file_out)
    file_name = file_name+'_pre.csv'
    path_pre  = cf.path_folder_pre + file_name
    df.to_csv(path_pre, mode='w', header=True, index=False)


def preprocessing_data_nhachoto(path_file ,istart, iend):
    df = pd.read_csv(path_file)
    # drop_row(df , istart, iend) # chỉ lấy từ dòng istart->iend
    file_name = utl.get_file_name_from_path(path_file)

    col_time    = cf.field_header_file_data[1]
    col_price   = cf.field_header_file_data[2]
    col_acreage = cf.field_header_file_data[3]
    col_address = cf.field_header_file_data[4]
    col_type    = cf.field_header_file_data[5]
    col_detail  = cf.field_header_file_data[6]

    str_data_get = file_name[file_name.rindex("_")+1:]
    data_get = datetime.strptime(str_data_get, "%d%m%Y")
    df[col_time] = df[col_time].map(lambda x: get_pre_month_year_from_str(data_get, x))
    df.drop(columns=col_detail, inplace=True)

    # Trường giá phòng : loại bỏ các ký tự không phải số và . + giá lớn hơn 0
    df[col_price] = df[col_price].map(lambda x: x[0: x.rindex("-")])
    df[col_price] = df[col_price].map(lambda x: x.replace("triệu/tháng", "").replace(",", "."))
    df[col_price] = df[col_price].map(lambda x: re.sub('[^0-9.]', '', x))
    df[col_price] = df[col_price].map(lambda x: float(x)*1000000 if float(x) < 100 else float(x)*1000)
    df[col_price] = df[col_price].map(lambda x: int(x))
    df = df[df[col_price] > 0]

    # Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    df[col_acreage] = df[col_acreage].map(lambda x: x[:-2])  # m2
    df[col_acreage] = df[col_acreage].map(lambda x: re.sub('[^0-9.]', '', x))

    file_name = file_name + '_pre.csv'
    path_pre = cf.path_folder_pre + file_name
    df.to_csv(path_pre, mode='w', header=True, index=False)
    return


def preprocessing_data_phongtro123(path_file):
    df = pd.read_csv(path_file)
    # drop_row(df , istart, iend) # chỉ lấy từ dòng istart->iend

    col_time    = cf.field_header_file_data[1]
    col_price   = cf.field_header_file_data[2]
    col_acreage = cf.field_header_file_data[3]
    col_address = cf.field_header_file_data[6]
    col_detail  = cf.field_header_file_data[7]

    #Trường thời gian :  loại col_timebỏ các ký tự không phải số và /
    df[col_time] = df[col_time].map(lambda x: re.sub('[^0-9//]', '', x))
    df[col_time] = df[col_time].map(lambda x: x[3:])

    #Trường giá phòng : loại bỏ các ký tự không phải số và . + giá lớn hơn 0
    df = df[df[col_price].str.contains("Thỏa thuận") == False]
    df[col_price] = df[col_price].map(lambda x: re.sub('[^0-9.]', '', x))
    df[col_price] = df[col_price].map(lambda x: float(x)*1000000 if float(x) < 100 else float(x)*1000)
    df[col_price] = df[col_price].map(lambda x: int(x))
    df = df[df[col_price] > 0]

    #Trường diện tích : bỏ m2 và bỏ các ký tự không phải là số
    df[col_acreage] = df[col_acreage].map(lambda x: x[:-2]) # m2
    df[col_acreage] = df[col_acreage].map(lambda x: re.sub('[^0-9.]', '', x))

    # Trường địa chỉ : bỏ từ Địa chỉ: trong dữ liệu
    df[col_address] = df[col_address].map(lambda x: x.replace("Địa chỉ: ", ""))

    # Trường chi tiết : bỏ từ các từ không cần thiết
    df[col_detail] = df[col_detail].str.lower()

    # Trường thuận tiện và tiện nghi: giuongtu,banghe,nonglanh,dieuhoa,tulanh,maygiat,tivi,bep,gacxep,thangmay,bancong,chodexe
    # for unit_cmp in cf.field_header_file_data_tiennghi:
    #     col_new = df[col_detail].map(lambda x: 1 if any(s in x for s in unit_cmp[1]) else 0)
    #     df.insert(len(df.columns) - 2, unit_cmp[0], col_new)

    #Trường có khép kín hay không:
    # df[col_type] = df[col_price]<= 1000000

    file_name = utl.get_file_name_from_path(path_file)
    file_name = file_name+'_pre.csv'
    path_pre  = cf.path_folder_pre + file_name

    df = df[cf.field_header_file_data]
    df.to_csv(path_pre, mode='w', header=True, index=False)
    return


def pre_detail_phongtro123(path_file):
    df = pd.read_csv(path_file)
    col_detail = cf.field_header_file_data[7]

    # Trường tiện nghi
    for unit_cmp in cf.field_header_file_data_tiennghi:
        col_new = df[col_detail].map(lambda x: 1 if any(s in x for s in unit_cmp[1]) else 0)
        df.insert(len(df.columns)-2,unit_cmp[0], col_new)

    df.to_csv(path_data_raw, mode='w', header=True, index=False)


def preprocessing_data_phongtro123_spider(path_file):
    df = pd.read_csv(path_file)
    # drop_row(df , istart, iend) # chỉ lấy từ dòng istart->iend

    # col_time = cf.field_header_file_spider[1]
    col_price   = cf.field_header_file_spider[3]
    col_acreage = cf.field_header_file_spider[4]
    col_address = cf.field_header_file_spider[8]
    col_detail  = cf.field_header_file_spider[9]

    #Trường thời gian :  loại col_timebỏ các ký tự không phải số và /
    # df[col_time] = df[col_time].map(lambda x: re.sub('[^0-9//]', '', x))
    # df[col_time] = df[col_time].map(lambda x: x[3:])

    #Trường giá phòng : loại bỏ các ký tự không phải số và . + giá lớn hơn 0
    df = df[df[col_price].str.contains("Thỏa thuận") == False]
    df[col_price] = df[col_price].map(lambda x: re.sub('[^0-9.]', '', x))
    df[col_price] = df[col_price].map(lambda x: float(x)*1000000 if float(x) < 100 else float(x)*1000)
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

    file_name = utl.get_file_name_from_path(path_file)
    file_name = file_name+'_pre.csv'
    path_pre  = folder_out_spider+file_name

    df = df[cf.field_header_file_data]
    df.to_csv(path_pre, mode='w', header=True, index=False)
    return


# Xử lý các cột trong dữ liệu thô sang file housepricedata_1.csv
def convert_rawdata_to_traindata(path_rawdata_in , path_traindata_out):
    df = pd.read_csv(path_rawdata_in)
    df.drop(['stt', 'chitiet', 'diachi','tiennghi','thuantien'], axis=1, inplace=True)
    df["thoigian"] = pd.to_datetime(df["thoigian"])
    df.insert(1, 'thang', df["thoigian"].dt.month)
    df.insert(0, 'nam', df["thoigian"].dt.year)
    df = df.drop(['thoigian'], axis=1)
    df.to_csv(path_traindata_out, index=False, header=True)


# Xử lý các cột trong dữ liệu thô sang file housepricedata_1.csv
def convert_rawdata_to_traindata2(path_rawdata_in , path_traindata_out):
    df = pd.read_csv(path_rawdata_in ,encoding='utf8')
    df.drop(['stt', 'chitiet', 'diachi','duong', 'phuong'], axis=1, inplace=True)
    df["thoigian"] = pd.to_datetime(df["thoigian"])
    df.insert(1, 'thang', df["thoigian"].dt.month)
    df.insert(0, 'nam', df["thoigian"].dt.year)
    df = df.drop(['thoigian'], axis=1)

    df['quan'].replace(cf.district_convert, inplace=True)
    df.to_csv(path_traindata_out, index=False, header=True)


def main():
    # load csv
    # pre_detail_phongtro123(path_data_raw_01)
    # preprocessing_data_phongtro123("data_raw/data_phongtro123_03052021.csv")


    # convert_rawdata_to_traindata(path_data_raw, path_data_train)
    convert_rawdata_to_traindata2(path_data_raw2, path_data_train2)
    # convert_rawdata_to_traindata2(path_data_raw2, path_data_imp)
    # convert_data_row(path_data_raw , path_data_out)
    return


# Hàm main
if __name__ == "__main__":
    main()