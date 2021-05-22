
path_folder_encoder    = 'encoders/'
path_folder_data_daily = 'data_daily/'
path_folder_url        = 'data_url/'
path_folder_data_raw   = 'data_raw/'
path_folder_pre        = "data_pre/"

field_header_file_scrap = ["stt", "thoigian", "giaphong", "dientich", "diachi","vido", "kinhdo", "chitiet"]
field_header_file_err = ["stt", "url"]
field_header_file_url = ["stt", "url"]

columns_name_data = ['nam','thang','giaphong','dientich','quan','vido','kinhdo','loai','drmd','kcdc',
                     'loaiwc','giuongtu','banghe','nonglanh','dieuhoa','tulanh','maygiat','tivi',
                     'bep','gacxep','thangmay','bancong','chodexe']

col_nam      = columns_name_data[0]
col_thang    = columns_name_data[1]
col_giaphong = columns_name_data[2]
col_dientich = columns_name_data[3]
col_quan     = columns_name_data[4]
col_vido     = columns_name_data[5]
col_kinhdo   = columns_name_data[6]
col_loai     = columns_name_data[7]
col_drmd     = columns_name_data[8]
col_kcdc     = columns_name_data[9]
col_loaiwc   = columns_name_data[10]
col_giuongtu = columns_name_data[11]
col_banghe   = columns_name_data[12]
col_nonglanh = columns_name_data[13]
col_dieuhoa  = columns_name_data[14]
col_tulanh   = columns_name_data[15]
col_maygiat  = columns_name_data[16]
col_tivi     = columns_name_data[17]
col_bep      = columns_name_data[18]
col_gacxep   = columns_name_data[19]
col_thangmay = columns_name_data[20]
col_bancong  = columns_name_data[21]
col_chodexe  = columns_name_data[22]




# Cấu hình bên API

cf_model_mlp    = {'path': "models/mlp_room_prediction.h5",    'reload': True, 'model': None}
cf_model_knn    = {'path': "models/knn_room_prediction.h5",    'reload': True, 'model': None}
cf_model_randf  = {'path': "models/randf_room_prediction.h5",  'reload': True, 'model': None}
cf_model_mlinear= {'path': "models/mlinear_room_prediction.h5",'reload': True, 'model': None}

cf_encoder      = {}

TF = 0 # Loại TrueFalse
RA = 1 # Loại Khoảng giá trị
LI = 2 # Loại liệt kê
NO = 3 # Không xử lý gì cả

api_params = [
         (col_nam      ,2020    ,RA,[2016,2200]                ,1,'Giá trị năm từ 2016 trở lên'),
         (col_thang    ,0       ,RA,[1,12]                     ,0,'Giá trị tháng [1->12]'),
         (col_dientich ,0.0     ,RA,[1,200]                    ,1,'Giá trị nằm trong khoảng [1->200]m^2'),
         (col_quan     ,'None'  ,NO,[1,200]                    ,1,'Giá trị quận không đúng'),
         (col_vido     ,0.0     ,RA,[20.93,21.09]              ,1,'Vi độ không thuộc nội thành Hà Nội'),
         (col_kinhdo   ,0.0     ,RA,[105.73,105.93]            ,1,'Kinh độ không thuộc nội thành Hà Nội'),
         (col_loai     ,'Nhacap',LI,["Nhacap","Nhatang",'Ccmn'],1,'Giá trị hợp lệ: [Nhà cấp/Nhà tầng/Ccmn]'),
         (col_drmd     ,0.0     ,RA,[1, 50]                    ,1,'Giá trị nằm trong khoảng [1->50]m'),
         (col_kcdc     ,0.0     ,RA,[1, 2000]                  ,1,'Giá trị nằm trong khoảng [1->2000]m'),
         (col_loaiwc   ,'KKK'   ,LI,["KKK", "Khepkin"]         ,1,'Giá trị hợp lệ : [KKK/Khép kín]'),
         (col_giuongtu ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_banghe   ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_nonglanh ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_dieuhoa  ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_tulanh   ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_maygiat  ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_tivi     ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_bep      ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_gacxep   ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_thangmay ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_bancong  ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
         (col_chodexe  ,0       ,TF,[]                         ,0,'Giá trị hợp lệ : [Có/Không]'),
]


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

district_convert ={
    'Thanh Xuân'  :"ThanhXuan" ,
    'Hoàng Mai'   :"HoangMai"  ,
    'Bắc Từ Liêm' :"BacTuLiem" ,
    'Cầu Giấy'    :"CauGiay"   ,
    'Hà Đông'     :"HaDong"    ,
    'Đống Đa'     :"DongDa"    ,
    'Nam Từ Liêm' :"NamTuLiem" ,
    'Hoàn Kiếm'   :"HoanKiem"  ,
    'Hai Bà Trưng':"HaiBaTrung",
    'Tây Hồ'      :"TayHo"     ,
    'Ba Đình'     :"BaDinh"    ,
    'Gia Lâm'     :"GiaLam"    ,
    'Long Biên'   :"LongBien"  ,
    'Thanh Trì'   :"ThanhTri"  ,
    'Hoài Đức'    :"HoaiDuc"
}