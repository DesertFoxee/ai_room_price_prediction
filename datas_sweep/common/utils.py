from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd


# phân tích cú pháp html lấy các trường dữ liệu tương ứng selectors
def parse_data_html_scrap(html_data_raw, arr_selectors):
    data_obj = [None] * len(arr_selectors)
    soup = BeautifulSoup(html_data_raw, features='html.parser')
    for index, key_selector in enumerate(arr_selectors):
        try:
            # chỉ sô đầu tiên nếu có sẽ chỉ phần tử thứ bao nhiêu trong list select ra được lấy
            index_selector = key_selector[0]
            if index_selector.isdigit():
                selector = key_selector[1::]
                str_data = soup.select(selector)[int(index_selector)].text
            else:
                str_data = soup.select_one(key_selector).text
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
def push_data_to_file(data_out, file_out, start_index, head=None, first=False):
    df = pd.DataFrame(data=data_out, columns=head)
    df.index += start_index
    df.index.name = "stt"
    df.to_csv(file_out, mode='a', header=first)
    first = False
