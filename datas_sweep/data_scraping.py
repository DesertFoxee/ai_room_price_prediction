import threading
import os
import pandas as pd
import common.config as cf
import common.utils as util

EVERY_TIME = 10
LIMIT_PUSH_DATA = 100
push_first_data = True
push_first_err = True

data_get = [
    # thời gian , giá phòng , diện tích  , địa chỉ , chi tiết
    ("urls_nhachoto", ('span[class*="imageCaptionText___"]', 'span[itemprop="price"]', 'span[itemprop="size"]',
                       'div[class*="address___"] > span', 'p[itemprop="description"]')),
    ("urls_phongtro123", ('3span[class="acreage"]', 'span[class="price"]', 'span[class="acreage"]',
                          'p[class="section-description"]', '.post-main-content > div[class="section-content"]'))
]


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
            threads = [threading.Thread(target=util.get_html_data_from_url_scrap,
                                        args=(url, arr_selectors, data_rooms, url_errs)) for url in urls]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if url_errs:  # Có lỗi xảy ra
                print("[ Done ] : " + str(EVERY_TIME - len(url_errs)) + "/" + str(EVERY_TIME)
                      + ". "+str(len(url_errs)) + " failed ! -> push to file :" + file_name_err_csv)
                util.push_data_to_file(url_errs, file_name_err_csv, start_index_err, push_first_data)
                start_index_err += len(url_errs)
            else:  # Hoàn thành không lỗi
                print("[  OK  ] !!!!!!")
                if len(data_rooms) >= LIMIT_PUSH_DATA:
                    util.push_data_to_file(data_rooms, file_name_data_csv,
                                           start_index_data, cf.data_fields_scraping, push_first_err)
                    start_index_data += len(data_rooms)
                    data_rooms.clear()
                    print("Over : Reset data ;)")
            print("========================================================")
        if data_rooms:
            util.push_data_to_file(data_rooms, file_name_data_csv,
                                   start_index_data, cf.data_fields_scraping)

# Hàm main
if __name__ == "__main__":
    main()
