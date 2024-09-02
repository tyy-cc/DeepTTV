import numpy as np 
from sklearn.linear_model import LinearRegression

def get_nontransit_a(
    m_t, # mass of transit planet
    m_nont, # mass of non-transit planet 
    m_star, # mass of star
    a_t, # semi-major axis of transit planet
    n_Rhill, # least number of Rhill needed from the transit planet
    locate_in_or_out, # non-transit is inside/ouside of the transit one or both: 'in', 'out', 'both' 
    has_range # select a range for the non-transit a or not
    ):
    DaysOfYear = 365.25
    X = ((m_t + m_nont)/(3*m_star))**(1/3)
    Rh_outer = X * a_t/(1 - n_Rhill/2*X)
    Rh_inner = X * a_t/(1 + n_Rhill/2*X)

    a_nontransit_outer_bound = a_t + n_Rhill*Rh_outer 
    a_nontransit_inner_bound = a_t - n_Rhill*Rh_inner 

    if has_range:
        a_10day = (m_star * (10/DaysOfYear)**2)**(1/3)
        a_100day = (m_star * (100/DaysOfYear)**2)**(1/3)

        a_nontransit_outer = np.random.uniform(a_nontransit_outer_bound, a_100day)
        a_nontransit_inner = np.random.uniform(a_10day, a_nontransit_inner_bound) 
        
        if locate_in_or_out == 'both':
            total_a_range = a_100day - a_nontransit_outer_bound + a_nontransit_inner_bound - a_10day
            p_outer = (a_100day - a_nontransit_outer_bound)/total_a_range 
            p_inner = (a_nontransit_inner_bound - a_10day)/total_a_range

            a_nontransit = np.random.choice([a_nontransit_outer, a_nontransit_inner], p=[p_outer, p_inner])

            return a_nontransit
        
        elif locate_in_or_out =='in':
            return a_nontransit_inner 

        elif locate_in_or_out =='out':
            return a_nontransit_outer 

    else:
        if locate_in_or_out == 'both':
            a_nontransit = np.random.choice([a_nontransit_outer_bound, a_nontransit_inner_bound])
            return a_nontransit

        elif locate_in_or_out == 'in':
            return a_nontransit_inner_bound

        elif locate_in_or_out == 'out':
            return a_nontransit_outer_bound

def get_ttv(epoch_list, transit_list):
    epoch_list = np.array(epoch_list)
    transit_list = np.array(transit_list)
    ttv_list = []
    # do linear regression
    do_reg = LinearRegression().fit(epoch_list.reshape((-1,1)),transit_list)
    Period_days = do_reg.coef_[0]
    reg_pred = do_reg.predict(epoch_list.reshape((-1,1)))

    for i in range(len(transit_list)):
        ttv_list.append(transit_list[i] - reg_pred[i])

    return ttv_list, Period_days

def get_tdv(epoch_list, dur_list):
    epoch_list = np.array(epoch_list)
    dur_list = np.array(dur_list)
    tdv_list = []
    # do linear regression
    do_reg = LinearRegression().fit(epoch_list.reshape((-1,1)), dur_list)
    dur_coef = do_reg.coef_[0]
    reg_pred = do_reg.predict(epoch_list.reshape((-1,1)))

    for i in range(len(dur_list)):
        tdv_list.append(dur_list[i] - reg_pred[i])

    return tdv_list, dur_coef, do_reg.intercept_

def get_frac_tdv(epoch_list, dur_list):
    epoch_list = np.array(epoch_list)
    dur_list = np.array(dur_list)
    tdv_list = []
    # do linear regression
    do_reg = LinearRegression().fit(epoch_list.reshape((-1,1)), dur_list)
    dur_coef = do_reg.coef_[0]
    reg_pred = do_reg.predict(epoch_list.reshape((-1,1)))

    for i in range(len(dur_list)):
        tdv_list.append((dur_list[i] - reg_pred[i])/reg_pred[i])

    return tdv_list, dur_coef, do_reg.intercept_

def get_ttv_duration(transit_data_list):

    start_record = False 
    days_in_year = 365.25
    duration_list = []
    transit_list = []
    epoch_list = []

    for one_transit_data in transit_data_list:

        # first transit starts 
        if not start_record and one_transit_data[1] == 'start':
            start_record = True 
            start = float(one_transit_data[2])
            epoch_list.append(float(one_transit_data[0]))

        elif start_record:

            if one_transit_data[1] == 'center':
                transit_list.append(float(one_transit_data[2])*days_in_year)
                
            elif one_transit_data[1] == 'end':
                end = float(one_transit_data[2])
                duration_list.append((end - start)*days_in_year)
                start = None

            elif one_transit_data[1] == 'start':
                start = float(one_transit_data[2])
                end = None 
                epoch_list.append(float(one_transit_data[0]))

    if len(transit_list) != 135 :
        print('Number of transit is not 135!')
         
        
    ttv_list, fitted_Period_days = get_ttv(epoch_list, transit_list) # unit in days
   
    tdv_list, dur_coef, dur_intercept = get_tdv(epoch_list, duration_list) # unit in days
    
    duration_list = np.array(duration_list)
    percent_tdv, _, _  = get_frac_tdv(epoch_list, duration_list)

    return ttv_list, duration_list, transit_list, fitted_Period_days, percent_tdv , tdv_list #, dur_coef, dur_intercept

def get_koi134_ttv_duration():
    transit_time = [-2013.751077, -1946.635564, -1879.452791, -1812.198971, -1744.867606, -1677.452204, -1609.96144,
                    -1542.38994, -1474.743281, -1407.036146, -1339.296122, -1271.548007, -1203.821742, -1136.146997,
                    -1001.001925, -933.5448008, -866.1671408, -798.8654835, -664.4853682, -597.3924326]
    # [153.252915, 220.366626, 287.547325, 354.802878, 422.135476,
    #            489.550584, 557.040524, 624.612593, 692.258616, 759.965327,
    #            827.705608, 895.453072, 963.179631, 1030.85548, 1166.000318,
    #            1233.457527, 1300.834013, 1368.136815, 1502.516773, 1569.609173] # days

    duration = [11.09, 11.25, 11.18, 11.29, 11.428, 11.33, 11.388, 11.41,
               11.366, 11.363, 11.394, 11.432, 11.341, 11.33, 11.313,
               11.11, 11.15, 11.02, 10.97, 10.79]
    # [0.477505, 0.487512, 0.47958, 0.484852, 0.496777, 0.486642,
    #             0.486482, 0.481162, 0.487988, 0.481378, 0.486276, 0.480615,
    #             0.477688, 0.495481, 0.475871, 0.487039, 0.472993, 0.467629,
    #             0.48068, 0.477505]
    if len(transit_time) != len(duration):
        print('error in get_koi134_ttv_duration!')

    epoch_transit = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,21,22])

    reg = LinearRegression().fit(epoch_transit.reshape((-1,1)), transit_time)
    reg_pred = reg.predict(epoch_transit.reshape((-1,1)))
    ttv_list = []
    for i in range(len(epoch_transit)):
        ttv_list.append(transit_time[i] - reg_pred[i])

    return ttv_list, duration, epoch_transit

def get_error_with_observation(ttv_list, duration_list, obs_ttv):
    obs_duration = [0.477505, 0.487512, 0.47958, 0.484852, 0.496777, 0.486642,
                0.486482, 0.481162, 0.487988, 0.481378, 0.486276, 0.480615,
                0.477688, 0.495481, 0.475871, 0.487039, 0.472993, 0.467629,
                0.48068, 0.477505]
    obs_epoch = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,21,22])
    
    ttv_error = []
    duration_error = []

    for i, epoch in enumerate(obs_epoch):
        
        ttv_error.append(abs(obs_ttv[i] - ttv_list[epoch - 1]))
        duration_error.append(abs(obs_duration[i] - duration_list[epoch - 1])) 

    return ttv_error, duration_error

def get_error_with_a_case(ttv_list, tdv_list):
    case_tdv = [-0.0014650384546008866, -0.0014929418497527935, -0.001193834002358729, 
                -0.0010433787416697093, -0.0005231284283396809, 6.858205724880051e-06, 
                0.0005333326602279254, 0.001027560156397378, 0.0013991634308023748, 
                0.0015046402762411981, 0.0016184410404099792, 0.0015717889886804892, 
                0.0013459476640423906, 0.0013198091189283723, 0.0006935073699345651, 
                -0.0003415986099628565, -0.0002756421575443868, -0.001195282263510733, 
                -0.0017162590897945584, -0.0017739453138554406]
    case_ttv = [0.7474935087021457, 0.4019087729372046, 0.07619394592634876, -0.17939073567620767, 
                -0.42598763085089786, -0.5167255121201606, -0.5938610952154022, -0.5118588903931709,
                  -0.4103822342095782, -0.22771953309677428, -0.027689797030006957, 0.17236442029411592,
                    0.378041417238137, 0.5257764655440269, 0.6565321626424065, 0.5875877127316471, 
                    0.41796234663183895, 0.18871872031240855, -0.4460266199828311, -0.8129374243840175]
    
    #obs_epoch = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,21,22])
    
    ttv_error = []
    tdv_error = []

    for i in range(len(case_ttv)):
        
        ttv_error.append(abs(case_ttv[i] - ttv_list[i]))
        tdv_error.append(abs(case_tdv[i] - tdv_list[i])) 

    return ttv_error, tdv_error

def get_ttv_wrt_134(transit_list):
    '''
        Here, the input 'transit_list' already ignore epoch 15 and epoch 20
    '''
    epoch_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,21,22])
    transit_list = np.array(transit_list[:20])
    ttv_list = []

    # do linear regression
    do_reg = LinearRegression().fit(epoch_list.reshape((-1,1)),transit_list)
    #Period_days = do_reg.coef_[0]
    reg_pred = do_reg.predict(epoch_list.reshape((-1,1)))
    Period_days = do_reg.coef_[0]
    for i in range(len(epoch_list)):
        ttv_list.append(transit_list[i] - reg_pred[i])
    return ttv_list, Period_days, transit_list

def get_tdv_wrt_134(duration_list):
    '''
        Here, the input 'duration_list' already ignore epoch 15 and epoch 20
    '''
    epoch_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,21,22])
    duration_list = np.array(duration_list[:20]) # duration_list = duration_list[epoch_list - 1]
    tdv_list = []
    # do linear regression
    do_reg = LinearRegression().fit(epoch_list.reshape((-1,1)), duration_list)
    #Period_days = do_reg.coef_[0]
    reg_pred = do_reg.predict(epoch_list.reshape((-1,1)))
    for i in range(len(epoch_list)):
        tdv_list.append(duration_list[i] - reg_pred[i])
    return tdv_list, do_reg.coef_[0], do_reg.intercept_, duration_list   

def get_all_diff_wrt_134(ttv_list, duration_list, tdv_list):
    duration_134 = np.array([11.09, 11.25, 11.18, 11.29, 11.428, 11.33, 11.388, 11.41,
               11.366, 11.363, 11.394, 11.432, 11.341, 11.33, 11.313,
               11.11, 11.15, 11.02, 10.97, 10.79])

    ttv_134 = np.array([0.7925414486396676, 0.3822295492821013, 0.03917764992456796, -0.23282724943305766, 
                        -0.42728714879058316, -0.5377100481480284, -0.5727709475054326, -0.5270958468631761, 
                        -0.40626174622047984, -0.2249516455781304, -0.010752544935485275, 0.21153755570685462, 
                        0.41197765634933603, 0.5608977569918352, 0.6543199582768011, 0.585619258919337, 
                        0.4374543595617979, 0.2132867602042552, -0.4582477385108632, -0.8911370378683614])

    tdv_134 = np.array([-0.298073696145126, -0.12384908037289222, -0.17962446460065706, -0.055399848828422193, 
                        0.09682476694381315, 0.01304938271604783, 0.0852739984882831, 0.121498614260517, 
                        0.09172323003275196, 0.10294784580498728, 0.14817246157722153, 0.20039707734945722, 
                        0.1236216931216898, 0.126846308893926, 0.13829554043839565, -0.050479843789370094, 
                        0.0037447719828662684, -0.11203061224490085, -0.1335813807004289, -0.29935676492819674])

    ttv_data = np.array(ttv_list)
    dur_data = np.array(duration_list)
    tdv_data = np.array(tdv_list)

    ttv_diff = ttv_data - ttv_134
    dur_diff = dur_data - duration_134 
    tdv_diff = tdv_data - tdv_134

    return ttv_diff, dur_diff, tdv_diff

