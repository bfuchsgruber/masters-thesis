import pandas as pd
from matplotlib import pyplot as plt

def read_input_from_iot_server(iot_server):
    """Read Excel input file and prepare URBS input dict.

    Reads the Excel spreadsheets that adheres to the structure shown in
    mimo-example.xlsx. Column titles in 'Demand' and 'SupIm' are split, so that
    'Site.Commodity' becomes the MultiIndex column ('Site', 'Commodity').

    Args:
        - filename: filename to Excel spreadsheets
        - year: current year for non-intertemporal problems

    Returns:
        a dict of up to 12 DataFrames
    """

    # Derive measurements from the IoT Server
    df = iot_server.get_df()
    # iot_server.plot()
    e3dc0power = df.E3DC0POWER.values
    inverter2 = df.INV2.values
    shelly_api_floor_power = df.SHELLY_API_FLOOR_POWER.values
    shelly_api_serverroom_power = df.SHELLY_API_SERVERROOM_POWER.values
    e3dc0soc = df.E3DC0SOC.values


    gl = []  # == "Global" Sheet in Excel file
    sit = [] # == "Site"
    com = [] # == "Commodity"
    pro = [] # == "Process"
    pro_com = [] # == Process-Commodity
    tra = [] # == "Transmission"
    sto = [] # == "Storage"
    dem = [] # == "Demand"
    sup = [] # == "SupIm"
    bsp = [] # == "Buy-Sell-Price"
    ds = [] # == "DSM"
    ef = [] # == "TimeVarEff"


    # prepare input data
    try:
        global_prop = pd.concat(gl, sort=False)
        site = pd.concat(sit, sort=False)
        commodity = pd.concat(com, sort=False)
        process = pd.concat(pro, sort=False)
        process_commodity = pd.concat(pro_com, sort=False)
        demand = pd.concat(dem, sort=False)
        supim = pd.concat(sup, sort=False)
        transmission = pd.concat(tra, sort=False)
        storage = pd.concat(sto, sort=False)
        dsm = pd.concat(ds, sort=False)
        buy_sell_price = pd.concat(bsp, sort=False)
        eff_factor = pd.concat(ef, sort=False)
    except KeyError:
        pass

    data = {
        'global_prop': global_prop,
        'site': site,
        'commodity': commodity,
        'process': process,
        'process_commodity': process_commodity,
        'demand': demand,
        'supim': supim,
        'transmission': transmission,
        'storage': storage,
        'dsm': dsm,
        'buy_sell_price': buy_sell_price.dropna(axis=1, how='all'),
        'eff_factor': eff_factor.dropna(axis=1, how='all')
    }

    # sort nested indexes to make direct assignments work
    for key in data:
        if isinstance(data[key].index, pd.MultiIndex):
            data[key].sort_index(inplace=True)

    print(data)
    return data
