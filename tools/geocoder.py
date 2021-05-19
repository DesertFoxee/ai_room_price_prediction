from opencage.geocoder import OpenCageGeocode
from pprint import pprint

key = '073a62ed72f6418181cb989e291d6ce6'
# geocoder = OpenCageGeocode(key)
#
# results = geocoder.reverse_geocode(21.047324, 105.779932)
# pprint(results)

key = '073a62ed72f6418181cb989e291d6ce6'
geocoder = OpenCageGeocode(key)

query = u'Hoàng Quốc Việt, Cầu Giấy, Hanoi'
results = geocoder.geocode(query)
print(u'%f,%f;%s;%s' % (results[0]['geometry']['lat'],
                        results[0]['geometry']['lng'],
                        results[0]['components']['country_code'],
                        results[0]['annotations']['timezone']['name']))