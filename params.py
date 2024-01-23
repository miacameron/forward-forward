

def get_params():

    par = dict()

    par['L1_par'] = dict()
    L1 = par['L1_par']
    L1['out_channels'] = 128
    L1['grid_dim'] = 4
    L1['rf_dim'] = 10
    L1['rf_channels'] = 1
    L1['rf_stride'] = 6


    par['L2_par'] = dict()
    L2 = par['L2_par']
    L2['out_channels'] = 220
    L2['grid_dim'] = 3
    L2['rf_dim'] = 2
    L2['rf_channels'] = 128
    L2['rf_stride'] = 1


    par['L3_par'] = dict()
    L3 = par['L3_par']
    L3['out_channels'] = 512
    L3['grid_dim'] = 2
    L3['rf_dim'] = 2
    L3['rf_channels'] = 220
    L3['rf_stride'] = 1


    par['thresh'] = 1
    par['lr'] = 0.01

    return par