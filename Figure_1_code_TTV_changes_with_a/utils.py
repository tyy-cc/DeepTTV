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

def get_ttv_noise(epoch_list, transit_list):

    epoch_list = np.array(epoch_list)
    transit_list = np.array(transit_list)
    noise = np.random.normal(loc=0.0, scale=1.0/60/24, size=transit_list.shape) #noise 1 min std
    transit_list += noise
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

def get_ttv_duration(transit_data_list, add_noise):

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
         
    if add_noise:
        ttv_list, fitted_Period_days = get_ttv_noise(epoch_list, transit_list) # unit in days
    else:
        ttv_list, fitted_Period_days = get_ttv(epoch_list, transit_list) # unit in days
   
    tdv_list, dur_coef, dur_intercept = get_tdv(epoch_list, duration_list) # unit in days
    
    duration_list = np.array(duration_list)
    percent_tdv, _, _  = get_frac_tdv(epoch_list, duration_list)

    return ttv_list, duration_list, transit_list, fitted_Period_days, percent_tdv , tdv_list #, dur_coef, dur_intercept


