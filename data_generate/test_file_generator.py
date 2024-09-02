import os
import shutil
import numpy as np
import math
from scipy.stats import uniform
from utils import get_nontransit_a
from zipfile import ZipFile
import sys 
import matplotlib.pyplot as plt 

FIXED_PARA = True 
USE_2013 = True
Folder_name = "kepler88_obs_2016_para2013_3_12"
pi = math.pi
DaysOfYear = 365.25

cur_path = os.getcwd()


num_test = 10000
locate_in_or_out_list = ['in','out'] # in, out
has_range = False

Earth_in_Solar = 3.0027e-6 # mass
Jupiter_in_Solar = 9.547919e-4 # mass
Solar_radius_in_km = 695700 # radius 
AU_in_km = 1.496e8 # AU

###################################
###################################
if USE_2013:
    mass_star = 0.956
    radius_star = 0.88
else:
    mass_star = 0.99 # solar mass
    radius_star = 0.897 # solar radius

e_upper = 0.1
e_lower = 0.01 

########## for transit planet #########
if USE_2013:
    period_transit = 10.95416/DaysOfYear
else:
    period_transit = 10.91647/DaysOfYear # days to year 

a_transit = (mass_star*period_transit**2)**(1/3) # AU


### boundary inclination for transit planet 
# incli_transit_bound = abs(np.arctan(radius_star*Solar_radius_in_km/(a_transit*AU_in_km))) 

########## for non-transit planet #########
mass_nontransit_upper = 10 * Jupiter_in_Solar #100 * Earth_in_Solar # M_Earth -> solar mass
mass_nontransit_lower = 10 * Earth_in_Solar # M_Earth -> solar mass 

count_test = 0

m_t_list = []
m_nont_list = []
e_t_list = []
e_nont_list = []
i_t_list = []
i_nont_list = []
omega_t_list = []
omega_nont_list = []
O_t_list = []
O_nont_list = []
mano_t_list = []
mano_nont_list = []
a_t_list = []
a_nont_list = []
a_nont_inner_bound_list = []
while count_test < num_test:

    locate_in_or_out = 'out'
    n_Rhill = 4 #np.random.uniform(3.5, 8.5)

    ############ randomly choose e ##################################
    if USE_2013:
        e_transit = 0.05593
        e_nontransit = np.random.uniform(0.01, 0.1)
    else:
        e_transit = 0.23578**2 + 0.0044**2
        # if FIXED_PARA:
        e_nontransit = 0.2392**2 + 0.0044**2
    # else: 
    #     e_nontransit = np.random.uniform(e_lower, e_upper)

    #################### mass #############################
    # if FIXED_PARA:
    if USE_2013:
        mass_transit = 8.7 * Earth_in_Solar
        mass_nontransit = np.random.uniform(mass_nontransit_lower, mass_nontransit_upper) 
    else:
        mass_transit = 0.0300 * Jupiter_in_Solar
        mass_nontransit = 0.674 * Jupiter_in_Solar

    #################### a ################################
    # if FIXED_PARA:
    if USE_2013:
        a_nontransit_inner_bound = get_nontransit_a(mass_transit, mass_nontransit, mass_star, a_transit, n_Rhill, locate_in_or_out, has_range)
        a_nontransit = np.random.uniform(max(a_nontransit_inner_bound, ((15/DaysOfYear)**2 * mass_star)**(1/3)), 5)
        # a_nontransit = ((22.3395/DaysOfYear)**2 * mass_star)**(1/3)
    else:
        a_nontransit = ((22.26492/DaysOfYear)**2 * mass_star)**(1/3)
    # else: 
    #     a_nontransit_inner_bound = get_nontransit_a(mass_transit, mass_nontransit, mass_star, a_transit, n_Rhill, locate_in_or_out, has_range)
    #     a_nontransit_outer_bound = (mass_star*(30/DaysOfYear)**2)**(1/3) # 30 day
    #     a_nontransit = np.random.uniform(a_nontransit_inner_bound, a_nontransit_outer_bound)

    #################### inclination ###################################
    incli_nontransit_bound = abs(np.arctan(radius_star*Solar_radius_in_km/(a_nontransit*AU_in_km)))
    # print(f'nontransit incli bound is {incli_nontransit_bound * 180/pi}')
    # if FIXED_PARA:
    if USE_2013:
        incli_transit = 0.945 * pi/180
        incli_nontransit = np.random.uniform(incli_nontransit_bound, (3.8 + 3*1.3)*pi/180)
    else:
        incli_transit =  abs(90 - 90.97) * pi/180 
        incli_nontransit = abs(90 - 93.15) * pi/180 
    # else:
    #     incli_transit_lower = (0.945 - 3 * 0.07) * pi/180
    #     incli_transit_upper = (0.945 + 3 * 0.074) * pi/180
    #     incli_transit = np.random.uniform(incli_transit_lower, incli_transit_upper)

         
    #     incli_nontransit = np.random.uniform(incli_nontransit_bound + 0.1*pi/180, (3.8 + 1.3*5)*pi/180) # 2013 paper
    

    ################# Omega  #########################
    
    if USE_2013:
        O_transit = 270 * pi/180
        O_nontransit = np.random.uniform(0, 2*pi) # np.random.uniform((264.1 - 11.6*3) * pi/180, (264.1 + 5*3) * pi/180)
    else:
        O_transit = 0 * pi/180 #np.random.uniform(-pi, pi)
        O_nontransit = -0.43* pi/180
    # else:
    #     O_nontransit = np.random.uniform(-pi, pi) 

    ################# omega ###########################
    l_transit_deg = 6.405 # mean longitude
    l_nontransit_deg = 252.9 #np.random.uniform(252.9 - 3*0.6, 252.9 + 3*0.94)
    g_transit_deg = 90.59 # pericenter longitude
    g_nontransit_deg = 270.76 #np.random.uniform(270.76 - 3*0.99, 270.76 + 3*1.82)

    if USE_2013:
        omega_transit = (g_transit_deg - O_transit*180/pi) * pi/180
        omega_nontransit = np.random.uniform(0, 2*pi) 
    else:
    # if FIXED_PARA:
        omega_transit =  np.arctan2(0.0044/np.sqrt(e_transit), -0.23578/np.sqrt(e_transit))
        omega_nontransit = np.arctan2(-0.0044/np.sqrt(e_nontransit), 0.2392/np.sqrt(e_nontransit))
    # else:
    #     omega_transit = np.random.uniform(omega_transit_lower, omega_transit_upper)
    #     omega_nontransit = np.random.uniform(-pi, pi)
    
    ###############  mean anomaly  #################
    
    if USE_2013:
        mean_ano_transit = (l_transit_deg - g_transit_deg) * pi/180
        mean_ano_nontransit = np.random.uniform(0, 2*pi)
    else:
    # if FIXED_PARA:
        n_transit = 2 * pi/10.91647
        n_nontransit = 2 * pi/22.26492
        ref_ang_transit = n_transit * (55.08069- 55.08069)
        ref_ang_nontransit = n_nontransit * (55.08069 - 61.353)
        mean_ano_transit = (ref_ang_transit - O_transit + omega_transit) % (2*pi)
        mean_ano_nontransit = ref_ang_nontransit - O_nontransit + omega_nontransit % (2*pi)
    # else:
    #     mean_ano_transit = 0 #np.random.uniform(-pi, pi)
    #     mean_ano_nontransit = np.random.uniform(-pi, pi)
    ##############################################

    count_test += 1

    m_t_list.append(mass_transit)
    m_nont_list.append(mass_nontransit)
    e_t_list.append(e_transit)
    e_nont_list.append(e_nontransit)
    i_t_list.append(incli_transit)
    i_nont_list.append(incli_nontransit)
    omega_t_list.append(omega_transit)
    omega_nont_list.append(omega_nontransit)
    O_t_list.append(O_transit)
    O_nont_list.append(O_nontransit)
    mano_t_list.append(mean_ano_transit)
    mano_nont_list.append(mean_ano_nontransit)
    a_t_list.append(a_transit)
    a_nont_list.append(a_nontransit)

m_t_list = np.array(m_t_list)
m_nont_list = np.array(m_nont_list)
e_t_list = np.array(e_t_list)
e_nont_list = np.array(e_nont_list)
i_t_list = np.array(i_t_list)
i_nont_list = np.array(i_nont_list)
omega_t_list = np.array(omega_t_list)
omega_nont_list = np.array(omega_nont_list)
O_t_list = np.array(O_t_list)
O_nont_list = np.array(O_nont_list)
mano_t_list = np.array(mano_t_list)
mano_nont_list = np.array(mano_nont_list)
a_t_list = np.array(a_t_list)
a_nont_list = np.array(a_nont_list)
a_nont_inner_bound_list = np.array(a_nont_inner_bound_list)

labels = [m_t_list, m_nont_list,
          e_t_list, e_nont_list,
          i_t_list, i_nont_list,
          omega_t_list, omega_nont_list,
          O_t_list, O_nont_list,
          mano_t_list, mano_nont_list,
           a_nont_list
        ]

label_names_plot = ['m_t (M_E)', 'm_nont (M_E)', 
               'e_t', 'e_nont', 
               'incli_t', 'incli_nont', 
               'omega_t', 'omega_nont', 
               'Omega_t', 'Omega_nont', 
               'Mean_ano_t', 'Mean_ano_nont', 
                'a_nont']

obs_vals = [8.7, 198.8, 
            0.05593, 0.05628,
            0.945, 3.8,
            90.59 - 270, 270.76 - 264.1,
            270, 264.1,
            6.405 - 90.59, 252.9 - 270.76,
            ((10.95416/DaysOfYear)**2 * mass_star)**(1/3), ((22.3395/DaysOfYear)**2 * mass_star)**(1/3)
            ]

fig, axs = plt.subplots(7, 2, figsize=(10, 25))
for i in range(len(label_names_plot)):
    label = labels[i]
    if i == 0 or i == 1:
        label /= Earth_in_Solar
    elif 4 <= i <= 11:
        label = label *180/np.pi
    
    axs[i//2][i%2].hist(label, bins=100)
    axs[i//2][i%2].set_title(label_names_plot[i])
    axs[i//2][i%2].axvline(x=obs_vals[i], color='r', linestyle='--', label='obs value')

plt.savefig(cur_path + '/TTV_files_ttv/test_parm_distribution_file_generator_3_12.png')
plt.close()


