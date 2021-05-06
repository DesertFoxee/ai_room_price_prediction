import threading
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import common.utils as util
import common.config as cf


EVERY_TIME = 10
LIMIT_PUSH_DATA = 10

setup_data = [
    [
        "https://phongtro123.com/tinh-thanh/ha-noi?orderby=moi-nhat&page={page_index}", # url lấy link mới nhất [spider]
        # selector lấy dữ liệu đường link và thời gian
        [
            '#left-col div[class="post-meta"]',   # thẻ đầu tiên bao lấy url
            'h3[class="post-title"] > a',         # thẻ trong lấy url [chú ý : là thẻ a]
            'time[class="post-time"]'             # thẻ lấy thời gian [chú ý : là thẻ lấy được text]
        ],
        # selector lấy dữ liệu chi tiết về phòng
        ('3.post-price-item', '0.post-price-item', '1.post-price-item',
                          '?data-address #__maps_content',
                          '?data-lat #__maps_content', '?data-long #__maps_content',
                          '.post-main-content > div[class="section-content"]'),

        ["phút", "giờ", "giây"],       # chuỗi trong thời gian được chấp nhận
        "phongtro123"
    ]
]

time_string_key = ["phút", "giờ"]
time_string_key_dis_run = ["hôm qua", "ngày" , "tuần"]

size_url_limit = 30
root_path  = util.get_root_path()


# lấy dữ liệu html thô và phân tích cú pháp lấy dữ liệu cần thiết
def get_html_data_from_url_spider(str_url, arr_selectors, arr_time_str_key):
    req = Request(str_url, headers={'User-Agent': 'Mozilla/5.0'})
    arr_new_url = []
    try:
        html_data_raw = urlopen(req, timeout=20).read()
        soup = BeautifulSoup(html_data_raw, features='html.parser')
        html_data_rows = soup.select(arr_selectors[0])
        for html_data_row in html_data_rows:
            str_link = html_data_row.select_one(arr_selectors[1])['href']
            str_time = html_data_row.select_one(arr_selectors[2]).text
            if any(ext in str_time for ext in arr_time_str_key):
                arr_new_url.append(str_link)
    except Exception as e:
        print(">> Parse error !!!")
    return arr_new_url


# Hàm lấy dữ liệu là link sang các trang chi tiết thuê phòng con spider
def data_crawler_spider(url, arr_selector, urls_data):
    html_data = util.get_html_data_from_url(url[1])
    if html_data is not None:
        data_obj = util.parse_html_data_to_obj(html_data, arr_selector)
        if data_obj is None:
            print('[Missing]: Parse from :' + url[1])
        else:
            util.inset_data(data_obj, url)
            urls_data.append(data_obj)
    else:
        print('[Missing]: Get data from :' + url[1])


def main():
    for page_infor in setup_data:

        # Thiết lập thông tin về page
        url_template       = page_infor[0]
        web_scan           = util.get_web_host_name_from_url(url_template)
        arr_selectors_url  = page_infor[1]   # nó luôn lớn hơn 3 và có cấu trúc
        arr_selectors_data = page_infor[2]   # nó luôn lớn hơn 3 và có cấu trúc
        arr_time_str_key   = page_infor[3]

        head_file = datetime.today().strftime('%d%m%Y')
        file_out_data = root_path + cf.path_folder_data_daily + head_file + "_" + page_infor[4] + "_data.csv"

        # Lấy các url mới nhất từng ngày
        print("[ START ] Scrapping : " + util.get_web_host_name_from_url(url_template))
        print("[*] Scanning new urls for the day...")
        data_urls = []
        index_page = 1
        while True:
            if len(data_urls) > size_url_limit:
                print("Overcame data limit !! -> Break ;(", end=" => ")
                break
            str_url = url_template.format(page_index=str(index_page))
            new_urls = get_html_data_from_url_spider(str_url, arr_selectors_url, arr_time_str_key)
            if not new_urls:
                print("Finished !!! -> Break :)", end=" => ")
                break
            for u in new_urls:
                data_urls.append(web_scan+u)
            index_page += 1

        print("[ Done ]: " + str(len(data_urls)) + " url")

        # Lấy dữ liệu từ url đã craping ở trên
        print("[*] Get data from url ...")
        data_rooms = []
        # Thêm phần header vào file mới
        util.push_header_to_file(file_out_data, cf.field_header_file_spider)

        for i_time in range(0, len(data_urls), EVERY_TIME):
            urls = []
            end_url = i_time + EVERY_TIME
            if end_url >= len(data_urls):
                end_url = len(data_urls)
            print("Scraping data from url : [" + str(i_time) + " -> " + str(end_url) + "]", end=" =>")
            for i in range(i_time, end_url, 1):
                urls.append([i, data_urls[i]])  # đẩy cả stt và url vào lúc tra cho dễ

            # Triển khai đa luồng scraping
            threads = [threading.Thread(target=data_crawler_spider, args=(url, arr_selectors_data, data_rooms))
                       for url in urls]
            util.run_thread(threads)

            print(" [OK] !!!!!!")
            if len(data_rooms) >= LIMIT_PUSH_DATA:
                print("=> Pushing " + str(len(data_rooms)) + " row to file " +file_out_data)
                util.push_data_to_exist_file(data_rooms, file_out_data)
                data_rooms.clear()
        if data_rooms:
            print("=> Pushing " + str(len(data_rooms)) + " row to file " + file_out_data)
            util.push_data_to_exist_file(data_rooms, file_out_data)
        print("[ END ] end scrapping : " + web_scan)


# Hàm main
if __name__ == "__main__":
    main()
