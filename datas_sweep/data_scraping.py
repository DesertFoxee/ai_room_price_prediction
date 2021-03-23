import threading
import csv
import os
import pandas as pd
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

EVERY_TIME = 10
LIMIT_PUSH_DATA = 100

data_head = ["giaphong", "dientich", "diachi", "chitiet"]
data_get = [
    # giaphong  , dien tich , o quan nao , chi tiet
    # ("urls_nhachoto", (
    #     'span[itemprop="price"]', 'span[itemprop="size"]', 'div[class*="address___"] > span',
    #     'p[itemprop="description"]')),
    # ("urls_phongtro123", ('span[class="price"]', 'span[class="acreage"]', 'p[class="section-description"]',
    #                       '.post-main-content > div[class="section-content"]'))
]


# phân tích cú pháp html lấy các trường dữ liệu tương ứng selectors
def parse_data_html_scrap(html_data_raw, arr_selectors):
    data_obj = [None] * len(arr_selectors)
    soup = BeautifulSoup(html_data_raw, features='html.parser')
    for index, selector_scan in enumerate(arr_selectors):
        try:
            str_data = soup.select_one(selector_scan).text
        except Exception as e:
            # Xuất hiện lỗi phân tích cú pháp
            return None
        data_obj[index] = str_data
    return data_obj


# lấy dữ liệu html thô và phân tích cú pháp lấy dữ liệu cần thiết
def get_html_data_from_url_scrap(str_url, arr_selectors, data_scraping, url_errs):
    req = Request(str_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html_data_raw = urlopen(req, timeout=20).read()
    except Exception as e:
        url_errs.append('[get failed]:' + str_url)
        return
    data_obj = parse_data_html_scrap(html_data_raw, arr_selectors)

    if data_obj is None:
        url_errs.append('[parse failed]:' + str_url)
    else:
        data_scraping.append(data_obj)


# Đẩy dữ liệu vào file chế độ ghi thêm
def push_data_to_file(data_out, file_out, start_index, head=None):
    df = pd.DataFrame(data=data_out, columns=head)
    df.index += start_index
    df.to_csv(file_out, mode='a', header=False)


def main():
    for data_page in data_get:
        # Lấy thông tin về web cần lấy link
        file_name_csv = data_page[0] + '.csv'
        file_name_err_csv = data_page[0] + '_err.csv'
        file_name_data_csv = data_page[0] + '_data.csv'
        arr_selectors = data_page[1]

        csv_data = pd.read_csv(file_name_csv)
        csv_data_urls = csv_data.drop(csv_data.columns[[0]], axis=1)
        data_urls = csv_data_urls[csv_data_urls.columns[0]].values

        data_rooms = []
        start_index_data = 0
        start_index_err = 0

        # xóa file lần đầu tiên: file dữ liệu, và file lỗi
        if os.path.exists(file_name_data_csv):
            os.remove(file_name_data_csv)
        if os.path.exists(file_name_err_csv):
            os.remove(file_name_err_csv)

        for i_time in range(0, len(data_urls), EVERY_TIME):
            urls = []
            for i in range(i_time, (i_time + EVERY_TIME), 1):
                print("[+] Scraping data from link : " + data_urls[i]);
                urls.append(data_urls[i])

            url_errs = []
            # Triển khai đa luồng scraping
            threads = [threading.Thread(target=get_html_data_from_url_scrap, args=(url, arr_selectors, data_rooms, url_errs))
                       for url in urls]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if url_errs:  # Có lỗi xảy ra
                print("[ Done ] : " + str(EVERY_TIME - len(url_errs)) + "/" + str(EVERY_TIME) \
                      + ". "+str(len(url_errs)) + " failed ! -> push to file :" + file_name_err_csv)
                push_data_to_file(url_errs, file_name_err_csv, start_index_err)
                start_index_err += len(url_errs)
            else:  # Hoàn thành không lỗi
                print("[  OK  ] !!!!!!")
                if len(data_rooms) >= LIMIT_PUSH_DATA:
                    push_data_to_file(data_rooms, file_name_data_csv, start_index_data, data_head)
                    start_index_data += len(data_rooms)
                    data_rooms.clear()
                    print("Over : Reset data ;)")
            print("========================================================")
        if data_rooms:
            push_data_to_file(data_rooms, file_name_data_csv, start_index_data, data_head)

# Hàm main
if __name__ == "__main__":
    main()
