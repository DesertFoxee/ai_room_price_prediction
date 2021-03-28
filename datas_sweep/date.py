import threading
import csv
import os
import pandas as pd
import re
from datetime import datetime ,timedelta
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

date_now = datetime.now()

data_convert_time = [
        ["phút", 0  ],
        ["giờ" , 0  ],
        ["hôm" , 0  ],

        ["ngày", 1  ],
        ["tuần", 7  ],
        ["tháng",30 ],
        ["năm" , 365]
    ]



#Tính thời gian sau khi trừ đi số ngày nhất định
def get_pre_time_today(day_sub):
    pre_month = date_now - timedelta(days=day_sub)
    str_pre_time = pre_month.strftime("%m/%Y")
    return str_pre_time



#lấy thời gian đơn vị
def get_pre_month_year_from_str(date_get, str_time):
    match = [arr_cv_time for arr_cv_time in data_convert_time if arr_cv_time[0] in str_time]
    if match:
        factor = re.sub("\\D", "", str_time)
        if not factor:
            factor = "0"
        day_sub = int(factor) * match[0][1]
        pre_time = datetime.strptime(date_get, "%d%m%Y") - timedelta(days=day_sub)
        str_date = pre_time.strftime("%m/%Y")
        return str_date
    return ""


def main():
    str_date_get = "26032021"
    str_time ="4 tuần"
    str_pre_date = get_pre_month_year_from_str(str_date_get, str_time)

    print(str_date_get +":" + str_pre_date +" => " +str_pre_date)


if __name__ == '__main__':
    main()
