from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
import ntpath
import urllib
import os


# Lấy địa chỉ chính từ url
def get_web_host_name_from_url(url):
    host_data = urllib.parse.urlparse(url)
    host_page = host_data.scheme + "://" + host_data.netloc
    return host_page


# Lấy dữ liệu html thô từ url
def get_html_data_from_url(url, err_msg=None):
    html_data = None
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html_data = urlopen(req, timeout=20).read()
    except Exception as ex:
        if err_msg is not None:
            err_msg = "[Error] get data from: " + url
    return html_data


# phân tích cú pháp html lấy các trường trường [href] tương ứng với selector
def parse_html_data_to_url(html_data_raw, selector):
    soup = BeautifulSoup(html_data_raw, features='html.parser')
    urls_data = []
    try:
        tag_link = soup.select(selector)
        for link in tag_link:
            str_link = link.get('href')
            urls_data.append(str_link)
    except Exception as ex:
        pass
    return urls_data


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


# Đẩy dữ liệu vào file chế độ ghi thêm từng mảng
def push_data_to_file(data_out, file_out, start_index, head=None, first=False):
    df = pd.DataFrame(data=data_out, columns=head)
    df.index += start_index
    df.index.name = "stt"
    df.to_csv(file_out, mode='a', header=first)
    first = False


# Đẩy dữ liệu vào file chế độ ghi mới sau tiền xử lý
def push_data_to_file(data_out, file_out, head=None):
    df = pd.DataFrame(data=data_out, columns=head)
    df.index.name = "stt"
    df.to_csv(file_out, mode='a', header=head)


# Đẩy dữ liệu vào file chế độ ghi mới sau tiền xử lý
def push_data_to_new_file(data_out, file_out, head=None):
    df = pd.DataFrame(data=data_out, columns=head)
    df.index.name = "stt"
    if head is not None:
        df.to_csv(file_out, mode='w', header=True)
    else:
        df.to_csv(file_out, mode='w', header=False)


# Lấy tên file từ path
def get_file_name_from_path(path):
    head, tail = ntpath.split(path)
    filename = os.path.splitext(tail)[0]
    return filename or ntpath.basename(head)
