import matplotlib.pyplot as plt
time = [0.0001,
0.00010155,
0.00010312,
0.00010471,
0.00010633,
0.00010797,
0.00010964,
0.00011134,
0.00011306,
0.00011481,
0.00011658,
0.00011838,
0.00012021,
0.00012207,
0.00012396,
0.00012587,
0.00012782,
0.0001298,
0.0001318,
0.00013384,
0.00013591,
0.00013801,
0.00014015,
0.00014231,
0.00014451,
0.00014675,
0.00014901,
0.00015132,
0.00015366,
0.00015603,
0.00015845,
0.00016089,
0.00016338,
0.00016591,
0.00016847,
0.00017108,
0.00017372,
0.00017641,
0.00017913,
0.0001819,
0.00018472,
0.00018757,
0.00019047,
0.00019342,
0.00019641,
0.00019944,
0.00020253,
0.00020566,
0.00020884,
0.00021206,
0.00021534,
0.00021867,
0.00022205,
0.00022549,
0.00022897,
0.00023251,
0.00023611,
0.00023976,
0.00024346,
0.00024723,
0.00025105,
0.00025493,
0.00025887,
0.00026287,
0.00026694,
0.00027106,
0.00027525,
0.00027951,
0.00028383,
0.00028822,
0.00029267,
0.0002972,
0.00030179,
0.00030646,
0.0003112,
0.00031601,
0.00032089,
0.00032585,
0.00033089,
0.00033601,
0.0003412,
0.00034648,
0.00035183,
0.00035727,
0.00036279,
0.0003684,
0.0003741,
0.00037988,
0.00038575,
0.00039172,
0.00039777,
0.00040392,
0.00041017,
0.00041651,
0.00042295,
0.00042949,
0.00043613,
0.00044287,
0.00044972,
0.00045667,
0.00046373,
0.0004709,
0.00047818,
0.00048557,
0.00049308,
0.0005007,
0.00050844,
0.0005163,
0.00052428,
0.00053239,
0.00054062,
0.00054897,
0.00055746,
0.00056608,
0.00057483,
0.00058372,
0.00059274,
0.0006019,
0.00061121,
0.00062066,
0.00063025,
0.00064,
0.00064989,
0.00065994,
0.00067014,
0.0006805,
0.00069102,
0.0007017,
0.00071255,
0.00072357,
0.00073475,
0.00074611,
0.00075765,
0.00076936,
0.00078125,
0.00079333,
0.0008056,
0.00081805,
0.0008307,
0.00084354,
0.00085658,
0.00086982,
0.00088327,
0.00089692,
0.00091079,
0.00092487,
0.00093917,
0.00095369,
0.00096843,
0.0009834,
0.00099861,
0.001014,
0.0010297,
0.0010456,
0.0010618,
0.0010782,
0.0010949,
0.0011118,
0.001129,
0.0011465,
0.0011642,
0.0011822,
0.0012005,
0.001219,
0.0012379,
0.001257,
0.0012764,
0.0012962,
0.0013162,
0.0013365,
0.0013572,
0.0013782,
0.0013995,
0.0014211,
0.0014431,
0.0014654,
0.0014881,
0.0015111,
0.0015344,
0.0015582,
0.0015822,
0.0016067,
0.0016315,
0.0016568,
0.0016824,
0.0017084,
0.0017348,
0.0017616,
0.0017889,
0.0018165,
0.0018446,
0.0018731,
0.0019021,
0.0019315,
0.0019613,
0.0019916,
0.0020224,
0.0020537,
0.0020855,
0.0021177,
0.0021504,
0.0021837,
0.0022174,
0.0022517,
0.0022865,
0.0023219,
0.0023578,
0.0023942,
0.0024312,
0.0024688,
0.002507,
0.0025457,
0.0025851,
0.0026251,
0.0026656,
0.0027069,
0.0027487,
0.0027912,
0.0028343,
0.0028782,
0.0029227,
0.0029678,
0.0030137,
0.0030603,
0.0031076,
0.0031557,
0.0032045,
0.003254,
0.0033043,
0.0033554,
0.0034073,
0.0034599,
0.0035134,
0.0035677,
0.0036229,
0.0036789,
0.0037358,
0.0037935,
0.0038522,
0.0039117,
0.0039722,
0.0040336,
0.004096,
0.0041593,
0.0042236,
0.0042889,
0.0043552,
0.0044225,
0.0044909,
0.0045603,
0.0046308,
0.0047024,
0.0047751,
0.0048489,
0.0049239,
0.005
]
data = [0.000000000080782,
0.000000000080095,
0.000000000079397,
0.000000000078688,
0.000000000077969,
0.000000000077238,
0.000000000076496,
0.000000000075743,
0.000000000074978,
0.000000000074201,
0.000000000073412,
0.000000000072611,
0.000000000071797,
0.000000000070971,
0.000000000070132,
0.00000000006928,
0.000000000068415,
0.000000000067537,
0.000000000066645,
0.000000000065739,
0.000000000064819,
0.000000000063885,
0.000000000062937,
0.000000000061974,
0.000000000060996,
0.000000000060003,
0.000000000058994,
0.00000000005797,
0.000000000056931,
0.000000000055875,
0.000000000054802,
0.000000000053714,
0.000000000052608,
0.000000000051485,
0.000000000050345,
0.000000000049187,
0.000000000048012,
0.000000000046818,
0.000000000045606,
0.000000000044375,
0.000000000043125,
0.000000000041855,
0.000000000040566,
0.000000000039257,
0.000000000037928,
0.000000000036578,
0.000000000035893,
0.000000000035349,
0.000000000034798,
0.000000000034238,
0.000000000033669,
0.000000000033091,
0.000000000032505,
0.000000000031909,
0.000000000031305,
0.000000000030691,
0.000000000030067,
0.000000000029434,
0.000000000028791,
0.000000000028138,
0.000000000027475,
0.000000000026801,
0.000000000026118,
0.000000000025423,
0.000000000024718,
0.000000000024003,
0.000000000023276,
0.000000000022537,
0.000000000021788,
0.000000000021027,
0.000000000020254,
0.000000000019469,
0.000000000018839,
0.000000000018466,
0.000000000018088,
0.000000000017703,
0.000000000017312,
0.000000000016916,
0.000000000016513,
0.000000000016104,
0.000000000015689,
0.000000000015267,
0.000000000014839,
0.000000000014404,
0.000000000013963,
0.000000000013514,
0.000000000013059,
0.000000000012597,
0.000000000012127,
0.00000000001165,
0.000000000011166,
0.000000000010825,
0.000000000010565,
0.000000000010301,
0.000000000010033,
0.0000000000097611,
0.0000000000094848,
0.0000000000092042,
0.0000000000089192,
0.0000000000086299,
0.000000000008336,
0.0000000000080376,
0.0000000000077347,
0.000000000007427,
0.0000000000071145,
0.0000000000068098,
0.0000000000066261,
0.0000000000064396,
0.0000000000062502,
0.0000000000060579,
0.0000000000058626,
0.0000000000056643,
0.000000000005463,
0.0000000000052585,
0.0000000000050509,
0.00000000000484,
0.0000000000046259,
0.0000000000044261,
0.0000000000042914,
0.0000000000041546,
0.0000000000040157,
0.0000000000038747,
0.0000000000037315,
0.000000000003586,
0.0000000000034384,
0.0000000000032884,
0.0000000000031361,
0.0000000000029903,
0.0000000000028896,
0.0000000000027873,
0.0000000000026834,
0.0000000000025779,
0.0000000000024708,
0.000000000002362,
0.0000000000022515,
0.0000000000021394,
0.0000000000020428,
0.0000000000019658,
0.0000000000018876,
0.0000000000018083,
0.0000000000017276,
0.0000000000016458,
0.0000000000015626,
0.0000000000014782,
0.0000000000014136,
0.0000000000013541,
0.0000000000012937,
0.0000000000012323,
0.00000000000117,
0.0000000000011067,
0.0000000000010425,
0.00000000000099525,
0.0000000000009491,
0.00000000000090223,
0.00000000000085464,
0.00000000000080631,
0.00000000000075723,
0.00000000000073523,
0.00000000000072512,
0.00000000000071485,
0.00000000000070442,
0.00000000000069383,
0.00000000000068307,
0.00000000000067215,
0.00000000000066107,
0.00000000000064981,
0.00000000000063837,
0.00000000000062676,
0.00000000000061497,
0.000000000000603,
0.00000000000059084,
0.00000000000057849,
0.00000000000056596,
0.00000000000055323,
0.0000000000005403,
0.00000000000052717,
0.00000000000051384,
0.00000000000050031,
0.00000000000048656,
0.0000000000004726,
0.00000000000045843,
0.00000000000044404,
0.00000000000042942,
0.00000000000041458,
0.00000000000039951,
0.00000000000038421,
0.00000000000036867,
0.00000000000035289,
0.00000000000033686,
0.00000000000032059,
0.00000000000030407,
0.00000000000028729,
0.00000000000027025,
0.00000000000025295,
0.00000000000023538,
0.00000000000021754,
0.00000000000019942,
0.00000000000018102,
0.00000000000016234,
0.00000000000015163,
0.00000000000014765,
0.00000000000014361,
0.00000000000013951,
0.00000000000013535,
0.00000000000013112,
0.00000000000012683,
0.00000000000012246,
0.00000000000011804,
0.00000000000011354,
0.00000000000010897,
0.00000000000010434,
0.000000000000099628,
0.000000000000094847,
0.000000000000089991,
0.000000000000085061,
0.000000000000080055,
0.000000000000074971,
0.000000000000069808,
0.000000000000064566,
0.000000000000059243,
0.000000000000053837,
0.000000000000048348,
0.000000000000042774,
0.000000000000037114,
0.000000000000032098,
0.000000000000030875,
0.000000000000029632,
0.00000000000002837,
0.000000000000027089,
0.000000000000025788,
0.000000000000024467,
0.000000000000023125,
0.000000000000021763,
0.00000000000002038,
0.000000000000018975,
0.000000000000017549,
0.0000000000000161,
0.000000000000014629,
0.000000000000013136,
0.000000000000011619,
0.000000000000010079,
8.5146E-15,
6.9264E-15,
6.5048E-15,
0.000000000000006159,
5.8079E-15,
5.4513E-15,
5.0893E-15,
4.7216E-15,
4.3483E-15,
3.9692E-15,
3.5842E-15,
3.1933E-15,
2.7963E-15,
2.3932E-15,
1.9839E-15
]
# 绘制折线图
plt.plot(time, data)

# 添加标题、X轴和Y轴标签
plt.title("Values over Time")
plt.xlabel("T/s")
plt.ylabel("dBz/dt")

# 显示图形
plt.show()

