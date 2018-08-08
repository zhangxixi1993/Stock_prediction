import matplotlib.pyplot as plt
import numpy as np

emd_mae=[0.018,0.0173,0.0166,0.0173,0.0139]
emd_rmse=[0.026,0.027,0.0241,0.0247,0.0226]

emd_rnn_mae=[0.0559,0.0245,0.0182,0.0179,0.0243]
emd_rnn_rmse=[0.0608,0.0299,0.0234,0.0231,0.0312]

plt.plot(emd_rmse,'-o',label='emd_rmse')
plt.plot(emd_rnn_rmse,'-*',label='emd_rnn_rmse')
plt.legend()
plt.show()