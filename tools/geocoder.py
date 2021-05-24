from opencage.geocoder import OpenCageGeocode
from pprint import pprint
import common.config as cf
import re

key = '073a62ed72f6418181cb989e291d6ce6'

replace_str = {
    'à': 'à',
    'ầ': 'ầ',
    'ấ': 'ấ',
}

replace_miss = {
    'ngọc thuỵ': 'long bien'
}


key_district =[
    'suburb',
    'town'
]
str_remove =["district"]


# Lấy huyện từ lat long
def get_district_coordinates(lat, long):
    # Lấy giá trị từ lat long giới hạn 2500/ngày api
    geocoder = OpenCageGeocode(key)
    results = geocoder.reverse_geocode(lat, long)
    data_district = ""
    for key_dis in key_district:
        try:
            data_district = results[0]['components'][key_dis]
            break
        except:
            print("[X] error : !!!!!!!! :" + key_dis +" + (" + str(lat) +","+ str(long)+")")
            continue

    # Tiền xử lý dữ liệu trước
    data_district = str(data_district).lower()
    for str_rm in str_remove:
        data_district = data_district.replace(str_rm, '')
    data_district    = re.sub(' +', ' ', data_district)
    for char_replace in replace_str:
        data_district = data_district.replace(char_replace, replace_str[char_replace])

    # Xử lý lỗi từ API
    for dis_replace in replace_miss:
        data_district = data_district.replace(dis_replace, replace_miss[dis_replace])

    data_district = data_district.strip()
    data_district_re = data_district.replace(" ", "")

    # Lấy giá trị trong map
    print(data_district + "++++" + data_district_re)
    district_name = None
    for district, value in cf.district_convert.items():
        if (data_district == district.lower()) or (data_district_re == value.lower()):
            district_name = value
            break
    return district_name


# Lấy thông tin lat long từ địa chỉ
def get_latlong_address(address):
    geocoder = OpenCageGeocode(key)
    results = geocoder.geocode(address)
    return results

# 21.023783719336993,105.84760665893556
# print(get_district_coordinates(21.023783719336993,105.84760665893556))


# geocoder = OpenCageGeocode(key)
#
# results = geocoder.reverse_geocode(21.047324, 105.779932)
# pprint(results)


# geocoder = OpenCageGeocode(key)


# results = geocoder.reverse_geocode(44.8303087, -0.5761911)
#
# query = u'Hoàng Quốc Việt, Cầu Giấy, Hanoi'
# results = geocoder.geocode(query)
# print(u'%f,%f;%s;%s' % (results[0]['geometry']['lat'],
#                         results[0]['geometry']['lng'],
#                         results[0]['components']['country_code'],
#                         results[0]['annotations']['timezone']['name']))