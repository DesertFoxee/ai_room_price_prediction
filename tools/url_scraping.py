import threading
import common.utils as utl
import common.config as cf


EVERY_TIME = 10
folder_out = 'data_url/'

# Dữ liệu [ web name + urlfetch + selector + Max page, file ra url] sử dụng cho fetch data url
cf_url_scraping = [
    # ("https://nha.chotot.com/ha-noi/thue-phong-tro?page=", 'li[class*="wrapperAdItem__"] > a',180, "urls_nhachoto.csv"),
    ("https://phongtro123.com/tinh-thanh/ha-noi?page=", 'h4[class="post-title"] > a', 200, "urls_phongtro123.csv")
]


# Hàm lấy dữ liệu là link sang các trang chi tiết thuê phòng
def url_crawler(url, selector, urls_data, count_err):
    host_name = utl.get_web_host_name_from_url(url)
    html_data = utl.get_html_data_from_url(url)
    if html_data is not None:
        urls = utl.parse_html_data_to_url(html_data, selector)
        if urls:
            urls = [host_name + url for url in urls]
            urls_data.extend(urls)
        else:
            count_err[0] += 1
    else:
        count_err[0] += 1


def main():
    for web_page in cf_url_scraping:
        # Lấy thông tin về web cần lấy link
        url_scan      = web_page[0]
        arr_selector  = web_page[1]
        total_page    = web_page[2]
        file_name_csv = folder_out + web_page[3]

        urls_data = []
        print("[**] Load data from link : " + utl.get_web_host_name_from_url(url_scan))
        for i_time in range(0, total_page, EVERY_TIME):
            urls_page = []
            start_page = i_time + 1
            end_page = i_time + EVERY_TIME
            if end_page >= total_page:
                end_page = total_page

            print("Load url from page : " + str(start_page) + " -> " + str(end_page), end=" =>")
            for i in range(start_page, end_page, 1):
                urls_page.append(url_scan + str(i))
            count_err = [0]

            # Triển khai đa luồng scraping
            threads = [threading.Thread(target=url_crawler, args=(url, arr_selector, urls_data, count_err)) for url in
                       urls_page]
            utl.run_thread(threads)

            if count_err[0] > 0:
                print(" [Done] : " + str(len(urls_page) - count_err) + "/" + len(urls_page) + " successful !")
            else:
                print(" [Done] : OK !!!!!!!!")
            print("=======================================================================")
        print("[OK]  Push data to file => " + file_name_csv)
        print("")
        utl.push_data_to_new_file(urls_data, file_name_csv, cf.field_header_file_url, index=True)


# Hàm main
if __name__ == "__main__":
    main()
