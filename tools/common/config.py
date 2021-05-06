
path_folder_encoder    = 'encoders/'
path_folder_data_daily = 'data_daily/'
path_folder_url        = 'data_url/'
path_folder_data_raw   = 'data_raw/'
path_folder_pre        = "data_pre/"

field_header_file_scrap = ["stt", "thoigian", "giaphong", "dientich", "diachi","vido", "kinhdo", "chitiet"]
field_header_file_err = ["stt", "url"]
field_header_file_url = ["stt", "url"]

columns_name_data = ['nam','thang','giaphong','dientich','vido','kinhdo','loai','drmd','kcdc',
                     'loaiwc','giuongtu','banghe','nonglanh','dieuhoa','tulanh','maygiat','tivi',
                     'bep','gacxep','thangmay','bancong','chodexe']

col_nam      = columns_name_data[0]
col_thang    = columns_name_data[1]
col_giaphong = columns_name_data[2]
col_dientich = columns_name_data[3]
col_vido     = columns_name_data[4]
col_kinhdo   = columns_name_data[5]
col_loai     = columns_name_data[6]
col_drmd     = columns_name_data[7]
col_kcdc     = columns_name_data[8]
col_loaiwc   = columns_name_data[9]
col_giuongtu = columns_name_data[10]
col_banghe   = columns_name_data[11]
col_nonglanh = columns_name_data[12]
col_dieuhoa  = columns_name_data[13]
col_tulanh   = columns_name_data[14]
col_maygiat  = columns_name_data[15]
col_tivi     = columns_name_data[16]
col_bep      = columns_name_data[17]
col_gacxep   = columns_name_data[18]
col_thangmay = columns_name_data[19]
col_bancong  = columns_name_data[20]
col_chodexe  = columns_name_data[21]


cf_model_mlp    = {'path': "models/mlp_room_prediction.h5",    'reload': True, 'model': None}
cf_model_knn    = {'path': "models/knn_room_prediction.h5",    'reload': True, 'model': None}
cf_model_randf  = {'path': "models/randf_room_prediction.h5",  'reload': True, 'model': None}
cf_model_mlinear= {'path': "models/mlinear_room_prediction.h5",'reload': True, 'model': None}

field_header_file_data = ["stt", "thoigian", "giaphong", "dientich","vido", "kinhdo", "diachi", "chitiet"]
field_header_file_data_tiennghi = [
                                    ['giuongtu', ['giường', 'tủ' ,'giuong','tu']],
                                    ['banghe',   ['bàn', 'ghế']],
                                    ['nonglanh', [' nl ','nong lanh','nóng lạnh','nonglanh']],
                                    ['dieuhoa' , ['dieu hoa','điều hòa','dieuhoa','máy lạnh','điều hoà']],
                                    ['tulanh' ,  [' tl ','tulanh','tu lanh','tủ lạnh']],
                                    ['maygiat',  [' nl ','maygiat','máy giặt','may giat' ]],
                                    ['tivi',     [' tv ','tivi']],
                                    ['bep',      ['bếp']],
                                    ['gacxep',   ['gác xép',' gx ','gác lửng']],
                                    ['thangmay', ['thang máy','thang may','thangmay']],
                                    ['bancong',  ['bancong','ban công']],
                                    ['chodexe',  ['để xe', 'de xe','gửi xe']]
                                    ]
field_header_file_data_thuantien = ['gacxep' , 'bancong' , 'chodexe' ,'camerachamvan','thangmay']
field_header_file_spider = ["stt","link" , "thoigian", "giaphong", "dientich","vido", "kinhdo","drmd", "diachi", "chitiet"]