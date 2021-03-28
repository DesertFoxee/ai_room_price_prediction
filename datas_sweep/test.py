import common.utils as ut
from bs4 import BeautifulSoup
import url_scraping as ws
#
# urls_data = []
# count_err = [0]
# ws.url_crawler("https://phongtro123.com/tinh-thanh/ha-noi?page=1", '4[class="post-title"] > a', urls_data,
#             count_err)
# print(count_err[0])
# for url in urls_data:
#     print(url)

# from opencage.geocoder import OpenCageGeocode
# from pprint import pprint
#
# key = '073a62ed72f6418181cb989e291d6ce6'
# # geocoder = OpenCageGeocode(key)
# #
# # results = geocoder.reverse_geocode(21.047324, 105.779932)
# # pprint(results)
#
# key = '073a62ed72f6418181cb989e291d6ce6'
# geocoder = OpenCageGeocode(key)
#
# query = u'Phường Khương Đình, Quận Thanh Xuân, Hà Nội'
# results = geocoder.geocode(query)
# print(u'%f,%f;%s;%s' % (results[0]['geometry']['lat'],
#                         results[0]['geometry']['lng'],
#                         results[0]['components']['country_code'],
#                         results[0]['annotations']['timezone']['name']))


data_raw = ut.get_html_data_from_url("https://phongtro123.com/phong-tro-khep-khep-kin-day-du-tien-nghi-nha-moi-xay-pho-bui-bui-xuong-trach-thanh-xuan-pr309654.html")

soup = BeautifulSoup(data_raw, features='html.parser')



selector_all = '?data-lat #__maps_content'

index_attr = selector_all.find(' ')
attr = selector_all[1:index_attr]
selector_sub = selector_all[index_attr+1:]


a = soup.select_one(selector_sub)[attr]
print(a)
