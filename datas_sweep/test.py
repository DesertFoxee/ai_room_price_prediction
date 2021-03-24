import common.utils as ut
import web_scraping as ws

urls_data = []
count_err = [0]
ws.url_crawler("https://phongtro123.com/tinh-thanh/ha-noi?page=1", '4[class="post-title"] > a', urls_data,
            count_err)
print(count_err[0])
for url in urls_data:
    print(url)
