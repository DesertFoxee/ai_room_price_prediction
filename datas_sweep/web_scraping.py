import threading

import pandas as pd
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from selenium import webdriver

WEB_SCAN = "https://batdongsan.com.vn/";
URL_SCAN = "https://nha.chotot.com/ha-noi/thue-phong-tro?page=";
# URL_SCAN = "https://batdongsan.com.vn/cho-thue-nha-tro-phong-tro-ha-noi/p"
URL_OUT_FILE = "linkdata_.csv";

EVERY_TIME = 10

# str_select = 'li[class*="wrapperAdItem__"] > a'
# str_select1 = 'li[class="post-item"] > figure > a'
data_header =["url"]

# Dữ liệu [ web name + urlfetch + selector + Max page ] sử dụng cho fetch data url
data_fetch = [
   # ("https://nha.chotot.com", "https://nha.chotot.com/ha-noi/thue-phong-tro?page=", 'li[class*="wrapperAdItem__"] > a',
   #  120, "urls_nhachoto.csv"),
   #  ("https://phongtro123.com", "https://phongtro123.com/tinh-thanh/ha-noi?page=", 'h4[class="post-title"] > a',
   #   120, "urls_phongtro123.csv")
]


def get_html_data_from_url(url, html_datas):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    data = urlopen(req, timeout=20).read()
    html_datas.append(data)


#     driver = webdriver.Chrome('./chromedriver.exe')
#     driver.get(URL_SCAN + str(1))
#     print(driver.title)

def main():
    for web_page in data_fetch:
        # Lấy thông tin về web cần lấy link
        web_scan = web_page[0]
        url_scan = web_page[1]
        selector_can = web_page[2]
        total_page = web_page[3]
        file_name_csv = web_page[4]
        urls_data = []

        print("[+] Load data from link : " + web_page[0]);
        for i_time in range(1, total_page, EVERY_TIME):
            html_data = []
            urls = []
            for i in range(1, EVERY_TIME + 1, 1):
                page_index = (i_time - 1) + i
                print(page_index)
                urls.append(url_scan + str(page_index))
            print("================================")

            # Triển khai đa luồng scraping
            threads = [threading.Thread(target=get_html_data_from_url, args=(url, html_data)) for url in urls]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Lấy đa ta dữ liêu
            for data_raw in html_data:
                soup = BeautifulSoup(data_raw, features='html.parser')
                tag_link = soup.select(selector_can)
                for link in tag_link:
                    str_link = link.get('href')
                    urls_data.append(web_scan + str_link)

        pd.DataFrame(data=urls_data).to_csv(file_name_csv, header=data_header, index_label="stt")


# Hàm main
if __name__ == "__main__":
    main()
