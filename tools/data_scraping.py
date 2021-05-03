import threading
import pandas as pd
import common.config as cf
import common.utils as utl
from datetime import datetime

EVERY_TIME = 10
LIMIT_PUSH_DATA = 100

folder_out = "data_raw"

data_get = [
    # thời gian , giá phòng , diện tích  , địa chỉ , chi tiết
    # ("urls_nhachoto", ('span[class*="imageCaptionText___"]', 'span[itemprop="price"]', 'span[itemprop="size"]',
    #                    'div[class*="address___"] > span', 'p[itemprop="description"]')),
    ("urls_phongtro123", ('3.post-price-item', '0.post-price-item', '1.post-price-item',
                          '?data-address #__maps_content',
                          '?data-lat #__maps_content', '?data-long #__maps_content',
                          '.post-main-content > div[class="section-content"]'))
]


# Hàm lấy dữ liệu là link sang các trang chi tiết thuê phòng
def data_crawler(url, arr_selector, urls_data, url_errs):
    html_data = utl.get_html_data_from_url(url[1])
    if html_data is not None:
        data_obj = utl.parse_html_data_to_obj(html_data, arr_selector)
        if data_obj is None:
            url_errs.append('[parse failed]: ' + url[1])
        else:
            data_obj.insert(0, url[0]) #insert stt tự từ url
            urls_data.append(data_obj)
    else:
        url_errs.append('[get failed]:' + url[1])


def main():
    for data_page in data_get:

        # Lấy thông tin về web cần lấy link
        tail_file = "_"+datetime.today().strftime('%d%m%Y') + ".csv"
        file_url_csv = data_page[0] + '.csv'
        file_name = data_page[0].replace("urls_", "")
        file_data_csv = folder_out + '/data_' + file_name + tail_file
        file_err_csv = folder_out + '/err_' + file_name + tail_file
        arr_selectors = data_page[1]
        print("[**] Loading url data from file : " + file_url_csv ,end= " => ")
        try:
            csv_data = pd.read_csv(file_url_csv)
            utl.push_header_to_file(file_data_csv, cf.field_header_file_scrap)
            utl.push_header_to_file(file_err_csv, cf.field_header_file_err)
            print("OK !!!!!!!")
        except Exception as ex:
            print("[Error] : Can't open file !!!")
            continue
        print("> Starting scarp from url....")
        data_urls = csv_data[csv_data.columns[1]].values
        data_stt = csv_data[csv_data.columns[0]].values
        data_rooms = []

        for i_time in range(0, len(data_urls), EVERY_TIME):
            urls = []
            end_url = i_time + EVERY_TIME
            if end_url >= len(data_urls):
                end_url = len(data_urls)
            print("Scraping data from url : [" + str(i_time) + " -> " + str(end_url) + "]", end=" =>")
            for i in range(i_time, end_url, 1):
                urls.append([data_stt[i], data_urls[i]]) # đẩy cả stt và url vào lúc tra cho dễ

            url_errs = []
            # Triển khai đa luồng scraping
            threads = [threading.Thread(target=data_crawler, args=(url, arr_selectors, data_rooms, url_errs))
                       for url in urls]
            utl.run_thread(threads)

            if url_errs:  # Có lỗi xảy ra
                print("[ Done ] : " + str(EVERY_TIME - len(url_errs)) + "/" + str(EVERY_TIME)
                      + ". "+str(len(url_errs)) + " failed ! -> push to file :" + file_err_csv)
                utl.push_data_to_exist_file(url_errs, file_err_csv)

            else:  # Hoàn thành không lỗi
                print(" [OK] !!!!!!")
                if len(data_rooms) >= LIMIT_PUSH_DATA:
                    utl.push_data_to_exist_file(data_rooms, file_data_csv)
                    data_rooms.clear()
                    print(" [OK] Over : Reset data ;)")
        if data_rooms:
            utl.push_data_to_exist_file(data_rooms, file_data_csv)
        print("=====================================================================")


# Hàm main
if __name__ == "__main__":
    main()
