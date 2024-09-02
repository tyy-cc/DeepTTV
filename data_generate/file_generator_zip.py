import os
import shutil
import numpy as np
import math
from scipy.stats import uniform
from utils import get_nontransit_a
from zipfile import ZipFile
import sys 
 
USE_2013 = True # Always True as using parameters based on Nesvorny et al., 2013
Folder_name = "Data_for_kepler88" # folder having nbody simulation docs
num_test = 4000 # number of simulation each folder
pi = math.pi
DaysOfYear = 365.25

cur_path = os.getcwd()

current_num = int(sys.argv[1])
cwd=cur_path + '/sim_folder_' + Folder_name +'/num_'+str(current_num)+'_xPos'
os.makedirs(cur_path + '/sim_folder_' + Folder_name, exist_ok=True)
os.mkdir(cwd)

os.chdir(cwd)
simulate_exe_path='../../Template/bin/simulate'

has_range = False

Earth_in_Solar = 3.0027e-6 # mass
Jupiter_in_Solar = 9.547919e-4 # mass
Solar_radius_in_km = 695700 # radius 
AU_in_km = 1.496e8 # AU

locate_in_or_out = 'out' # non-transiting planet locates inside "in" or outside "out" w.r.t. the transiting planet
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

########## for non-transit planet #########
mass_nontransit_lower = 10 * Earth_in_Solar # M_Earth -> solar mass 
mass_nontransit_upper = 10 * Jupiter_in_Solar #100 * Earth_in_Solar # M_Earth -> solar mass

shutil.copy('../../Template/sample/Test_mass/target_time.json','./target_time.json') # copy target file

count_test = 0

while count_test < num_test:

    n_Rhill = 4 #np.random.uniform(3.5, 8.5)

    ############ randomly choose e ##################################
    if USE_2013:
        e_transit = 0.05593
        e_nontransit = np.random.uniform(0.01, 0.1)
    else:
        e_transit = 0.23578**2 + 0.0044**2
        e_nontransit = 0.2392**2 + 0.0044**2

    #################### mass #############################
    if USE_2013:
        mass_transit = 8.7 * Earth_in_Solar
        mass_nontransit = np.random.uniform(mass_nontransit_lower, mass_nontransit_upper) 
    else:
        mass_transit = 0.0300 * Jupiter_in_Solar
        mass_nontransit = 0.674 * Jupiter_in_Solar

    #################### a ################################
    if USE_2013:
        a_nontransit_inner_bound = get_nontransit_a(mass_transit, mass_nontransit, mass_star, a_transit, n_Rhill, locate_in_or_out, has_range)
        a_nontransit = np.random.uniform(max(a_nontransit_inner_bound, ((15/DaysOfYear)**2 * mass_star)**(1/3)), 5)  
    else:
        a_nontransit = ((22.26492/DaysOfYear)**2 * mass_star)**(1/3)


    #################### inclination ###################################
    incli_nontransit_bound = abs(np.arctan(radius_star*Solar_radius_in_km/(a_nontransit*AU_in_km)))

    if USE_2013:
        incli_transit = 0.945 * pi/180
        incli_nontransit = np.random.uniform(incli_nontransit_bound, (3.8 + 3*1.3)*pi/180)
    else:
        incli_transit =  abs(90 - 90.97) * pi/180 
        incli_nontransit = abs(90 - 93.15) * pi/180 
    
    ################# Omega  #########################
    if USE_2013:
        O_transit = 270 * pi/180
        O_nontransit = np.random.uniform(0, 2*pi) 
    else:
        O_transit = 0 * pi/180 #np.random.uniform(-pi, pi)
        O_nontransit = -0.43* pi/180

    ################# omega ###########################
    l_transit_deg = 6.405 # mean longitude
    l_nontransit_deg = 252.9 #np.random.uniform(252.9 - 3*0.6, 252.9 + 3*0.94)
    g_transit_deg = 90.59 # pericenter longitude
    g_nontransit_deg = 270.76 #np.random.uniform(270.76 - 3*0.99, 270.76 + 3*1.82)

    if USE_2013:
        omega_transit = (g_transit_deg - O_transit*180/pi) * pi/180
        omega_nontransit = np.random.uniform(0, 2*pi) 
    else:
        omega_transit =  np.arctan2(0.0044/np.sqrt(e_transit), -0.23578/np.sqrt(e_transit))
        omega_nontransit = np.arctan2(-0.0044/np.sqrt(e_nontransit), 0.2392/np.sqrt(e_nontransit))
    
    ###############  mean anomaly  #################
    if USE_2013:
        mean_ano_transit = (l_transit_deg - g_transit_deg) * pi/180
        mean_ano_nontransit = np.random.uniform(0, 2*pi)
    else:
        n_transit = 2 * pi/10.91647
        n_nontransit = 2 * pi/22.26492
        ref_ang_transit = n_transit * (55.08069- 55.08069)
        ref_ang_nontransit = n_nontransit * (55.08069 - 61.353)
        mean_ano_transit = (ref_ang_transit - O_transit + omega_transit) % (2*pi)
        mean_ano_nontransit = ref_ang_nontransit - O_nontransit + omega_nontransit % (2*pi)
    ##############################################

    count_test += 1

    with open('file_names.txt', 'a') as file_name_write:
        file_name_write.write('num_'+str(current_num)+'_Test_' + str(count_test) + ' /mt_' + str(mass_transit) + '_mnont_' + str(mass_nontransit) # mass
                                + '_et_'+str(e_transit)+'_enont_'+str(e_nontransit) # e
                                + '_it_'+str(incli_transit)+'_inont_'+str(incli_nontransit) # i
                                + '_at_' + str(a_transit)+ '_anont_'+ str(a_nontransit) # a
                                + '_omegat_' + str(omega_transit) + '_omeganont_' + str(omega_nontransit) # omega
                                + '_Ot_'+str(O_transit)+ '_Onont_'+ str(O_nontransit) # Omega
                                + '_Meant_' + str(mean_ano_transit)+'_Meannont_' + str(mean_ano_nontransit) +'\n') # Mean anomaly



    with open('../../Template/sample/Test_mass/init_system.json') as init_file: # modify the init file

        new_file = open('init_system.json', 'w')
   
        for i, line in enumerate(init_file):

            if i == 4:
                new_file.write('        { "name":"transit", "rigid": false, "mass": '+ str(mass_transit)+', "orbital_elements": ['+ str(a_transit)+', '+ str(e_transit) +', '+ str(incli_transit) +', '+ str(O_transit) +', '+ str(omega_transit)+', ' + str(mean_ano_transit)+ '], "center_mass": '+ str(mass_star) +', "radius":24109.36},' + '\n') # radius doesn't matter as we use point mass
            elif i == 5:
                new_file.write('        { "name":"nontransit", "rigid": false, "mass": '+ str(mass_nontransit)+', "orbital_elements": ['+ str(a_nontransit)+', '+ str(e_nontransit) +', '+ str(incli_nontransit) +', '+ str(O_nontransit)+', ' + str(omega_nontransit) + ', ' + str(mean_ano_nontransit)+ '], "center_mass": '+ str(mass_star) +', "radius": 69911}' + '\n')
            elif i == 1:
                new_file.write('    "system_name":'+ ' "num_'+str(current_num)+'_Test_' + str(count_test) +'",' +'\n')
            elif i == 3:
                new_file.write('        { "name":"TSun", "rigid": false, "mass":'+ str(mass_star) +', "position": [0, 0, 0], "velocity": [0, 0, 0], "radius":'+str(radius_star*Solar_radius_in_km)+'},'+'\n')
            else:
                new_file.write(line)
        new_file.close()
    os.system(simulate_exe_path+' ./')
    
    if os.path.exists('current_system.json'):
        os.remove('current_system.json') # won't be exists, even when the code stops early, remove here just for checking
        print('hahaha current_system')
    

    current_txt='num_'+str(current_num)+'_Test_'+str(count_test)+'.txt'

    if os.path.exists('init_system_posvel.json'):
        os.remove('init_system_posvel.json') # exists if stops early


    current_txt='num_'+str(current_num)+'_Test_'+str(count_test)+'.txt'
    if os.path.exists(current_txt):
        with ZipFile('reuslt.zip', 'a') as zip:
            zip.write(current_txt, current_txt)

        os.remove(current_txt)

    os.remove('init_system.json')
os.remove('target_time.json')



