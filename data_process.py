import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
# import impyute as impy
from fancyimpute import KNN, SimpleFill
# import keras.backend as K
import gc

# version 1.0
#one hot
patients = []
# onehot
dfd = pd.read_csv('/home/leew/icu_data/lw_icu.csv')
dfdays = pd.read_csv('/home/leew/icu_data/lw_icu_hd.csv')
dfdcol = dfd.columns
for hid in dfdays.values[:, 0]:
    temv = dfd[dfd['hadm_id'] == hid].values
    temdf = pd.DataFrame(temv, columns=dfd.columns)
    temdf['curr_careunit'] = ((temdf['curr_careunit']).ffill()).bfill()
    temdf['curr_wardid'] = ((temdf['curr_wardid']).ffill()).bfill()
    for rec in temdf.values:
        patients.append(rec)
    print(hid)
finalv = np.array(patients)
print(finalv.shape)
finaldf = pd.DataFrame(finalv, columns=dfd.columns)
finaldf = finaldf.drop(columns=['startdate'])
finaldf = pd.get_dummies(finaldf, columns=['gender', 'religion', 'language_', 'marital_status', 'ethnicity', 'curr_careunit', 'curr_wardid'])
finaldf.to_csv('/home/leew/icu_data/lw_icu_onehot.csv', index=False)
finaldf = pd.read_csv('/home/leew/icu_data/lw_icu_onehot.csv')
for col in finaldf.columns:
    if(finaldf[col].isnull().any()==True):
        print(col)


#impute
dfd = pd.read_csv('/home/leew/icu_data/lw_icu_onehot.csv')
dfdv = np.array(dfd.values, dtype=float)
dfdcolumns = dfd.columns
partsize = 5000
partsnum = int(dfdv.shape[0]/partsize)
dfdshape = dfdv.shape
dfdimp = []
dfdfinal = []
del dfd
gc.collect()
print('sum %d parts' % partsnum)
for i in range(partsnum):
    part = []
    if i+1 == partsnum:
        part = dfdv[i*partsize:]
        # print(i*partsize, 'end')
    else:
        part = dfdv[i*partsize:(i+1)*partsize]
        # print(i*partsize, (i+1)*partsize)
    part = KNN(k=10, verbose=False).fit_transform(part)
    # part = impy.fast_knn(part, k=10)
    for row in part:
        dfdimp.append(row)
    dfdfinal = dfdimp
    dfdfinal = np.array(dfdfinal)
    dfdimpute = pd.DataFrame(dfdfinal, columns=dfdcolumns)
    for col in dfdimpute.columns:
        if dfdimpute[col].isnull().sum() != 0:
            print('error', col, dfdimpute[col].isnull().sum())
    np.save('/home/leew/icu_data/lw_icu_knnimpute.npy', dfdfinal)
    dfdimpute.to_csv('/home/leew/icu_data/lw_icu_knnimpute.csv', index=False)
    print('part %d done' % (i+1))
    print('current shape:', dfdfinal.shape)
print('impute finish!!!')





# read features
# k = 0
# df = pd.read_csv('/home/leew/icu_data/lw_icu_knnimpute.csv')
# print(df.values.shape)
# for i in df.columns:
#     # if(df[i].isnull().any()==True):
#     print(k, i)
#
#     k += 1


# make dataset x_y ward
x, y = [], []
dfhd = pd.read_csv('/home/leew/icu_data/lw_icu_hd.csv')
dfdimpute = pd.read_csv('/home/leew/icu_data/lw_icu_knnimpute.csv')
for i in range(dfdimpute.columns.shape[0]):
    print(i, dfdimpute.columns[i])
for hid in dfhd.values[:, 0]:
    x.append(dfdimpute[dfdimpute['hadm_id'] == hid].values[:, 1:177])
    y.append(dfdimpute[dfdimpute['hadm_id'] == hid].values[:, 177:])
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)
np.save('/home/leew/icu_data/x_ward.npy', x)
np.save('/home/leew/icu_data/y_ward.npy', y)


# make dataset x_y care
x, y = [], []
dfhd = pd.read_csv('/home/leew/icu_data/lw_icu_hd.csv')
dfdimpute = pd.read_csv('/home/leew/icu_data/lw_icu_knnimpute.csv')
for i in range(dfdimpute.columns.shape[0]):
    print(i, dfdimpute.columns[i])
for hid in dfhd.values[:, 0]:
    x.append(np.concatenate((dfdimpute[dfdimpute['hadm_id'] == hid].values[:, 1:172], dfdimpute[dfdimpute['hadm_id'] == hid].values[:, 177:]), axis=-1))
    y.append(dfdimpute[dfdimpute['hadm_id'] == hid].values[:, 172:177])
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)
np.save('/home/leew/icu_data/x_careunit.npy', x)
np.save('/home/leew/icu_data/y_careunit.npy', y)


# print(np.concatenate(([[1, 2], [3, 4]], [[5, 6], [7, 8]]), axis=-1))


# # # make dataset seqsdown
# dfhd = pd.read_csv('/home/leew/icu_data/lw_icu_hd.csv')
# dfdimpute = pd.read_csv('/home/leew/icu_data/lw_icu_knnimpute.csv')
# print(dfhd.values.shape)
# print(dfdimpute.values.shape)
# y, dualseqs = [], []
# for i in range(dfdimpute.columns.shape[0]):
#     print(i, dfdimpute.columns[i])
# for hid in dfhd.values[:, 0]:
#     tem = dfdimpute[dfdimpute['hadm_id'] == hid].values
#     hdays = tem.shape[0]
#     # print(tem.shape, hdays)
#     if tem.shape[0] != hdays:
#         print('error')
#     # print(hid, len)
#     hy = tem[:, 177:]
#     for i in hy:
#         if sum(i) == 0:
#             print('all 0')
#     y_ = []
#     for j in range(hdays):
#         if j + 1 == hdays:
#             y_.append(np.zeros(hy.shape[1], dtype=int))
#         else:
#             y_.append(hy[j + 1])
#     y.append(np.array(y_[:-1]))
#     dualseqs.append(tem[:-1, 1:])
# dualseqs = np.array(dualseqs)
# y = np.array(y)
# print(y[0].shape)
# print(dualseqs[0].shape)
# sumdays = 0
# for p in y:
#     # print(p.shape)
#     sumdays += p.shape[0]
#     for d in p:
#         if sum(d) == 0:
#             print('all 0')
# print('sumdays:', sumdays)
# np.save('/home/leew/icu_data/dualseqs_down_ward.npy', dualseqs)
# np.save('/home/leew/icu_data/seqy_down_ward.npy', y)






###### features ######
# 0 hadm_id
# 1 agap
# 2 albu
# 3 blood_glucose
# 4 bica
# 5 bilirut
# 6 blood_oxygen_saturation
# 7 body_temperature
# 8 calc
# 9 chlor
# 10 co
# 11 creatin
# 12 diastolic_blood_pressure
# 13 fio2
# 14 glascow_coma_scale
# 15 heart_rate
# 16 hematt
# 17 hemogt
# 18 interrat
# 19 lacta
# 20 partialt
# 21 pco
# 22 po
# 23 pota
# 24 ph
# 25 respiratory_rate
# 26 sodi
# 27 spo2
# 28 systolic_blood_pressure
# 29 ureanit
# 30 wbcet
# 31 age
# 32 weight
# 33 height
# 34 hospital_expire_flag
# 35 days
# 36 gender_f
# 37 gender_m
# 38 religion_7thdayadventist
# 39 religion_buddhist
# 40 religion_catholic
# 41 religion_christianscientist
# 42 religion_episcopalian
# 43 religion_greekorthodox
# 44 religion_hebrew
# 45 religion_hindu
# 46 religion_jehovah'switness
# 47 religion_jewish
# 48 religion_muslim
# 49 religion_notspecified
# 50 religion_other
# 51 religion_protestantquaker
# 52 religion_romanianeast.orth
# 53 religion_unitarian-universalist
# 54 religion_unobtainable
# 55 language__**sh
# 56 language__**t
# 57 language__**to
# 58 language__*amh
# 59 language__*ara
# 60 language__*arm
# 61 language__*be
# 62 language__*ben
# 63 language__*bos
# 64 language__*bul
# 65 language__*bur
# 66 language__*can
# 67 language__*cdi
# 68 language__*chi
# 69 language__*dut
# 70 language__*far
# 71 language__*fil
# 72 language__*fu
# 73 language__*ful
# 74 language__*guj
# 75 language__*hun
# 76 language__*ibo
# 77 language__*khm
# 78 language__*leb
# 79 language__*man
# 80 language__*mor
# 81 language__*nep
# 82 language__*per
# 83 language__*phi
# 84 language__*rom
# 85 language__*spa
# 86 language__*tam
# 87 language__*tel
# 88 language__*toi
# 89 language__*toy
# 90 language__*urd
# 91 language__*yid
# 92 language__*yor
# 93 language__alba
# 94 language__amer
# 95 language__arab
# 96 language__beng
# 97 language__camb
# 98 language__cant
# 99 language__cape
# 100 language__engl
# 101 language__ethi
# 102 language__fren
# 103 language__gree
# 104 language__hait
# 105 language__hind
# 106 language__ital
# 107 language__japa
# 108 language__kore
# 109 language__laot
# 110 language__mand
# 111 language__pers
# 112 language__poli
# 113 language__port
# 114 language__ptun
# 115 language__russ
# 116 language__serb
# 117 language__soma
# 118 language__span
# 119 language__taga
# 120 language__thai
# 121 language__turk
# 122 language__urdu
# 123 language__viet
# 124 marital_status_divorced
# 125 marital_status_lifepartner
# 126 marital_status_married
# 127 marital_status_separated
# 128 marital_status_single
# 129 marital_status_unknown(default)
# 130 marital_status_widowed
# 131 ethnicity_americanindian/alaskanative
# 132 ethnicity_americanindian/alaskanativefederallyrecognizedtribe
# 133 ethnicity_asian
# 134 ethnicity_asian-asianindian
# 135 ethnicity_asian-cambodian
# 136 ethnicity_asian-chinese
# 137 ethnicity_asian-filipino
# 138 ethnicity_asian-japanese
# 139 ethnicity_asian-korean
# 140 ethnicity_asian-other
# 141 ethnicity_asian-thai
# 142 ethnicity_asian-vietnamese
# 143 ethnicity_black/african
# 144 ethnicity_black/africanamerican
# 145 ethnicity_black/capeverdean
# 146 ethnicity_black/haitian
# 147 ethnicity_caribbeanisland
# 148 ethnicity_hispanic/latino-centralamerican(other)
# 149 ethnicity_hispanic/latino-colombian
# 150 ethnicity_hispanic/latino-cuban
# 151 ethnicity_hispanic/latino-dominican
# 152 ethnicity_hispanic/latino-guatemalan
# 153 ethnicity_hispanic/latino-honduran
# 154 ethnicity_hispanic/latino-mexican
# 155 ethnicity_hispanic/latino-puertorican
# 156 ethnicity_hispanic/latino-salvadoran
# 157 ethnicity_hispanicorlatino
# 158 ethnicity_middleeastern
# 159 ethnicity_multiraceethnicity
# 160 ethnicity_nativehawaiianorotherpacificislander
# 161 ethnicity_other
# 162 ethnicity_patientdeclinedtoanswer
# 163 ethnicity_portuguese
# 164 ethnicity_southamerican
# 165 ethnicity_unabletoobtain
# 166 ethnicity_unknown/notspecified
# 167 ethnicity_white
# 168 ethnicity_white-brazilian
# 169 ethnicity_white-easterneuropean
# 170 ethnicity_white-othereuropean
# 171 ethnicity_white-russian
# 172 curr_careunit_CCU
# 173 curr_careunit_CSRU
# 174 curr_careunit_MICU
# 175 curr_careunit_SICU
# 176 curr_careunit_TSICU
# 177 curr_wardid_2.0
# 178 curr_wardid_3.0
# 179 curr_wardid_4.0
# 180 curr_wardid_7.0
# 181 curr_wardid_8.0
# 182 curr_wardid_9.0
# 183 curr_wardid_11.0
# 184 curr_wardid_12.0
# 185 curr_wardid_13.0
# 186 curr_wardid_14.0
# 187 curr_wardid_15.0
# 188 curr_wardid_17.0
# 189 curr_wardid_18.0
# 190 curr_wardid_23.0
# 191 curr_wardid_24.0
# 192 curr_wardid_25.0
# 193 curr_wardid_27.0
# 194 curr_wardid_28.0
# 195 curr_wardid_29.0
# 196 curr_wardid_31.0
# 197 curr_wardid_32.0
# 198 curr_wardid_33.0
# 199 curr_wardid_36.0
# 200 curr_wardid_39.0
# 201 curr_wardid_40.0
# 202 curr_wardid_41.0
# 203 curr_wardid_44.0
# 204 curr_wardid_45.0
# 205 curr_wardid_48.0
# 206 curr_wardid_49.0
# 207 curr_wardid_50.0
# 208 curr_wardid_51.0
# 209 curr_wardid_52.0
# 210 curr_wardid_54.0
# 211 curr_wardid_55.0
# 212 curr_wardid_57.0
