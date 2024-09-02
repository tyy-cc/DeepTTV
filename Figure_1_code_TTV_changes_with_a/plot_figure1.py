import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import pickle
import os 
from sklearn.linear_model import LinearRegression
import sys 
from utils import * 
from statsmodels.nonparametric.smoothers_lowess import lowess

plt.rc('font', size=13) 
cwd = os.getcwd()
cwd = str(cwd)
DaysOfYear = 365.25
mass_star = 0.956

label_names_plot = ['m_t (M_E)', 'm_nont (M_E)', 
               'e_t', 'e_nont', 
               'incli_t', 'incli_nont', 
               'omega_t', 'omega_nont', 
               'Omega_t', 'Omega_nont', 
               'Mean_ano_t', 'Mean_ano_nont', 
               'a_nont']

label_names = ['m_t', 'Non-transit Mass', 
               'e_t', 'Non-transit e', 
               'incli_t', 'incli_nont', 
               'omega_t', 'omega_nont', 
               'Omega_t', 'Omega_nont', 
               'Mean_ano_t', 'Mean_ano_nont', 
               'a_nont']

label_plot_nont_only = {'Mass (M_E)': 1, 'Eccentricity': 3,
                        'Inclination (deg.)': 5, '$\omega$ (deg.)': 7,
                        '$\Omega$ (deg.)': 9, 'M (deg.)': 11, 'Semi-major axis (AU)':12}
obs_vals = [8.7, 198.8, 
            0.05593, 0.05628,
            0.945, 3.8,
            90.59 - 270, 270.76 - 264.1,
            270, 264.1,
            6.405 - 90.59, 252.9 - 270.76 + 360,
            ((22.3395/DaysOfYear)**2 * mass_star)**(1/3)]
Earth_in_Solar = 3.0027e-6 # mass


def read_data_from_GRIT_result(file_name) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    with open(cwd + '/TTV_files_ttv/'+ file_name, 'rb') as fileo:
        datao = pickle.load(fileo)
    fileo.close()

    print('reading data from pkl')

    epoch = np.array(list(range(1, 135 + 1)))
    ttvs = datao[0]
    tdvs = datao[1]
    # per_tdv = datao[1]
    duration = datao[2]
    average_dur = np.mean(duration, axis=1)
    # per_tdv = (duration - average_dur.reshape(-1,1))/average_dur.reshape(-1,1)
    transit_time = datao[3]
    fitted_P = datao[4]
    labels = np.array(datao[-1])

    return epoch, ttvs, labels, transit_time, tdvs, duration, fitted_P


# TTV obtained by GRIT using parameters from Nesvorny etal., 2013
# min to hour
N_2013_ttv =  np.array([583.47679047,  539.27488332,  496.74379878,  447.34695048, 401.41949783,  348.06958045,  298.64591889,  242.15884832,
                        188.94536405,  129.92797046,   72.74325193,   11.88396001, -49.09164788, -110.74867831, -174.68353622, -235.5043758 ,
                        -300.67580919, -358.26207986, -421.90351924, -473.05385612, -531.32182533, -572.19417149, -620.3422636 , -646.86614707,
                        -679.68378321, -688.2161838 , -700.79603518, -688.8763428 , -677.56547251, -644.56246057, -607.83972332, -555.24141734,
                        -494.16458225, -425.45794082, -343.52104982, -263.75579994, -166.33791448,  -81.54322698,   24.74593051,  108.11497281,
                        216.06691091,  291.67842045,  393.96660622,  456.15931152, 545.86250237,  590.27453685,  661.51850552,  685.65497881,
                        734.36597538,  737.9496079 ,  762.51006085,  747.352488,
                        748.75201643,  718.22528538,  699.58302394,  657.79952625,
                        623.48875647,  574.52107041,  529.20912566,  476.53455886,
                        424.47869217,  370.67341099,  315.29988785,  261.99203899,
                        205.78369668,  153.7692629 ,   98.34972896,   47.79616081,
                        -5.86877605,  -55.14695146, -106.48321796, -154.75475768,
                        -203.31078929, -250.66598228, -295.85458764, -341.9635308 ,
                        -382.85172863, -426.78277316, -461.95206586, -502.05887919,
                        -529.58549629, -563.52257024, -581.05454958, -605.9248782 ,
                        -610.98368713, -623.62103771, -614.04442882, -611.51022513,
                        -585.94516542, -566.08422648, -524.46991792, -486.46516393,
                        -430.27256932, -374.97983554, -307.22977504, -237.21639044,
                        -162.24631621,  -81.58418678,   -4.66823737,   81.4529093 ,
                        154.48471014,  240.16636475,  303.4385382 ,  382.47890216,
                        430.62455197,  497.00249012,  525.82891779,  574.30640426,
                        581.48127703,  608.38425399,  593.88689082,  597.81318906,
                        563.92951734,  546.06783393,  496.82962721,  460.6028284 ,
                        400.94520408,  351.16001907,  286.04677223,  227.92685363,
                        161.66938226,  100.04154566,   35.96077589,  -25.22279712,
                        -84.88023811, -142.71814788, -196.70071196, -249.28798224,
                        -297.11483361, -343.34251203, -385.05759707, -424.38374936,
                        -460.2781904 , -492.47955864, -522.86476599])/60


epoch, ttvs, labels, transit_time, per_tdv, duration, fitted_P = read_data_from_GRIT_result('kepler88_obs_2016_para2013_Data_7_19_forDiff_corrected_9_2.pkl')

ttvs_in_hr = ttvs * 24

for i in range(1000):
    plt.plot(epoch, ttvs_in_hr[i], 'o', alpha = 0.3)
plt.plot(epoch, N_2013_ttv, 'o', label='N2013')
plt.ylabel('TTV')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.savefig('plot_ttv_corrected_9_2.png', bbox_inches='tight')
plt.close()

chi_sqr = np.sum((ttvs_in_hr - N_2013_ttv)**2/np.abs(N_2013_ttv), axis=1)
print(chi_sqr.shape)

a_diff = labels[:, -1] - obs_vals[-1]
plt.plot(a_diff, chi_sqr, 'o', markersize=3)
plt.yscale('log')
plt.xlabel('$a_i$ - a (AU)')
plt.ylabel('$\chi^2$ (Hours)')
plt.savefig('plot_ttv_diff_wrt_a_corrected_9_2.png', bbox_inches='tight', dpi=300)
plt.close()
