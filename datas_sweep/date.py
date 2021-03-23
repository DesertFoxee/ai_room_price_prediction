import threading
import csv
import os
import pandas as pd
import re
from datetime import datetime ,timedelta
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

date_now = datetime.now()


#Tính thời gian sau khi trừ đi số ngày nhất định
def get_pre_time_today(day_sub):
    pre_month = date_now - timedelta(days=day_sub)
    str_pre_time = pre_month.strftime("%m/%Y")
    return str_pre_time

data_convert_time = [
        ["phút", 0  ],
        ["giờ" , 0  ],
        ["hôm" , 0  ],

        ["ngày", 1  ],
        ["tuần", 7  ],
        ["tháng",30 ],
        ["năm" , 365]
    ]

def main():
    str_url = "https://nha.chotot.com/ha-noi/quan-cau-giay/thue-phong-tro/69649327.htm"

    req = Request(str_url, headers={'User-Agent': 'Mozilla/5.0'})
    html_data_raw = urlopen(req, timeout=20).read()
    soup = BeautifulSoup(html_data_raw, features='html.parser')

    str_data_time = soup.select_one('span[class*="imageCaptionText___"]').text
    int_day_init =0
    str_day_int = ""

    #lấy thời gian đơn vị
    for time_checks in data_convert_time:
        if str_data_time.find(time_checks[0]) != -1:
            int_day_init = time_checks[1]
            str_day_int = time_checks[0]
            break;

    #Lấy hệ số của nó
    factor = re.sub("\D", "", str_data_time)
    if not factor :
        factor = "0"

    print(factor +" : " +  str_day_int)

    day_sub = int(factor) * int_day_init
    str_time = get_pre_time_today(day_sub)
    print(str_time)

if __name__ == '__main__':
    main()
