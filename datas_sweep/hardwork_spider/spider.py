import threading
import pandas as pd
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import data_scraping as scrap

EVERY_TIME = 10
LIMIT_PUSH_DATA = 10

setup_data = [
    [
        "https://nha.chotot.com",
        "https://nha.chotot.com/ha-noi/thue-phong-tro?page={page_index}&sp=0",
        [
            'li[class*="wrapperAdItem__"]',   # thẻ đầu tiên bao lấy url
            'a',                              # thẻ trong lấy url [chú ý : là thẻ a]
            'span[class*="item"]'             # thẻ lấy thời gian [chú ý : là thẻ lấy được text]
        ],
        ["phút", "giờ"],              # chuỗi trong thời gian được chấp nhận
        ["hôm qua", "ngày", "tuần"],  # không lấy và kết thúc trong thẻ thời gian
        "nhachotot"
    ],
    [
        "https://phongtro123.com",
        "https://phongtro123.com/tinh-thanh/ha-noi?orderby=moi-nhat&page={page_index}",
        [
            'div[class="post-meta"]',           # thẻ đầu tiên bao lấy url
            'h4[class="post-title"] > a',       # thẻ trong lấy url [chú ý : là thẻ a]
            'time[class="post-time"]'],         # thẻ lấy thời gian [chú ý : là thẻ lấy được text]
        ["phút", "giờ"],               # chuỗi trong thời gian được chấp nhận
        ["hôm qua", "ngày" , "tuần"],  # không lấy và kết thúc trong thẻ thời gian
        "phongtro123"
     ]
]

time_string_key = ["phút", "giờ"]
time_string_key_dis_run = ["hôm qua", "ngày" , "tuần"]

size_url_limit = 50
url_links =[]


# phân tích cú pháp html lấy các trường dữ liệu tương ứng selectors
def parse_data_html_spider(html_data_raw, arr_selectors, arr_time_str_key, arr_time_str_key_dis):
    soup = BeautifulSoup(html_data_raw, features='html.parser')
    arr_new_url = []
    try:
        html_data_rows = soup.select(arr_selectors[0])
        for html_data_row in html_data_rows:
            str_link = html_data_row.select_one(arr_selectors[1])['href']
            str_time = html_data_row.select_one(arr_selectors[2]).text
            if any(ext in str_time for ext in arr_time_str_key_dis):
                break
            if any(ext in str_time for ext in arr_time_str_key):
                arr_new_url.append(str_link)
    except Exception as e:
        print("Parse error !!!")
    return arr_new_url


# lấy dữ liệu html thô và phân tích cú pháp lấy dữ liệu cần thiết
def get_html_data_from_url_spider(str_url, arr_selectors, arr_time_str_key, arr_time_str_key_dis, err_urls):
    req = Request(str_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html_data_raw = urlopen(req, timeout=20).read()
    except Exception as e:
        err_urls.append('[get failed]:' + str_url)
        return
    new_urls = parse_data_html_spider(html_data_raw, arr_selectors,arr_time_str_key,arr_time_str_key_dis)
    return new_urls


def push_data_to_file(data_out, file_out, start_index, head=None):
    df = pd.DataFrame(data=data_out, columns=head)
    df.index += start_index
    df.to_csv(file_out, mode='a', header=False)


def main():
    for page_infor in setup_data:

        # Thiết lập thông tin về page
        web_scan    = page_infor[0]
        url_template = page_infor[1]
        arr_selectors = page_infor[2]   # nó luôn lớn hơn 3 và có cấu trúc
        arr_time_str_key = page_infor[3]
        arr_time_str_key_dis = page_infor[4]
        file_out_data = page_infor[5] + '_data.csv'
        file_out_url_err = page_infor[5] + '_err.csv'


        # lấy các url mới nhất từng ngày
        print("[ START ] Scrapping : " + web_scan)
        print("[*] Get new url...")
        data_urls = []
        index_page = 1
        is_running = True
        while is_running:
            if index_page >= 5 or len(data_urls) > size_url_limit:
                break
            url = url_template.format(page_index=str(index_page))
            new_urls = get_html_data_from_url_spider(url, arr_selectors, arr_time_str_key ,arr_time_str_key_dis)
            for u in new_urls:
                data_urls.append(web_scan+u)
            index_page += 1
        print("[ Done ]: "+ str(len(data_urls)) + " url")
        print("[*] Get data from url ...")

        # lấy dữ liệu từ url trên
        data_rooms = []
        start_index_data = 0
        start_index_err = 0
        for i_time in range(0, len(data_urls), EVERY_TIME):
            urls = []
            for i in range(i_time, (i_time + EVERY_TIME), 1):
                print("[+] Scraping data from link : " + data_urls[i]);
                urls.append(data_urls[i])

            url_errs = []
            # Triển khai đa luồng scraping
            threads = [threading.Thread(target= scrap.get_html_data_from_url_scrap, args=(url, arr_selectors, data_rooms, url_errs))
                       for url in urls]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if url_errs:  # Có lỗi xảy ra
                print("[ Done ] : " + str(EVERY_TIME - len(url_errs)) + "/" + str(EVERY_TIME) \
                      + ". "+str(len(url_errs)) + " failed ! -> push to file :" + file_out_url_err)
                push_data_to_file(url_errs, file_out_url_err, start_index_err)
                start_index_err += len(url_errs)
            else:  # Hoàn thành không lỗi
                print("[  OK  ] !!!!!!")
                if len(data_rooms) >= LIMIT_PUSH_DATA:
                    push_data_to_file(data_rooms, file_out_data, start_index_data, scrap.data_head)
                    start_index_data += len(data_rooms)
                    data_rooms.clear()
                    print("Over : Reset data ;)")
            print("========================================================")
        if data_rooms:
            push_data_to_file(data_rooms, file_out_data, start_index_data, scrap.data_head)
        print("[ END ] end scrapping : " + web_scan)
# Hàm main
if __name__ == "__main__":
    main()
