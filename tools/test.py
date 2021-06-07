import api as ap
import common.utils as utl
import common.config as cf
import numpy as np
import pandas as pd
path_data_train2 = 'roomdata2.csv'
path_data_train3 = 'roomdata2_sh.csv'

# root_path = utl.get_root_path()
#
# obj_param = ap.standardize_room_param(ap.test_param)
# model_param_1D = np.array(list(obj_param.values()))
# model_param = model_param_1D.reshape(1, -1)
#
# model = utl.load_model(root_path + cf.cf_model_randf['path'])
#
# # print(model_param)
# print(model.predict(model_param))

df = pd.read_csv(path_data_train3)
# utl.show_distribution(df[cf.col_giaphong].values, "Phân phối giá phòng", "Giá phòng", "Tần suất")
# utl.show_distribution(df[cf.col_dientich].values, "Phân phối diện tích", "Diện tích", "Tần suất")
utl.show_distribution_y(df,'quan', "Thống kê bản ghi theo quận", "Tần suât", "Quận")

# utl.load_all_encoder()
#
# a = ap.month_encoder(1)
# print(a)