from .input import split_columns
import pandas as pd
import numpy as np

def read_input_direct(supim_w:np.array, 
                      demand_w:np.array, 
                      soc_init:float=0.95, 
                      price_elec_buy:float=0.30, 
                      price_elec_sell:float=0.07,
                      demand_prediction_w:float=0.0,
                      supim_prediction_w:float=0.0,
                      demand_scale_factor:float=1.0,
                      supim_scale_factor:float=1.0,
                      kwp:float=18.0, 
                      year=2024):
    
    
    """Simulate an Excel input file and prepare URBS input dict.

    Reads the Excel spreadsheets that adheres to the structure shown in
    mimo-example.xlsx. Column titles in 'Demand' and 'SupIm' are split, so that
    'Site.Commodity' becomes the MultiIndex column ('Site', 'Commodity').

    Args:
        - supim_w: PV power in unit watt (NOT NORMED, since this will be done by the function)
        - demand_w: Power demand in unit watt
        - soc_init: SoC of the BSS at the beginning of the optimization in [0...1]
        - price_elec_buy: Price for buying 1 kWh energy in [EUR/kWh]
        - price_elec_sell: Price for feeding 1 kWh energy into the grid in [EUR/kWh]
        - demand_prediction_w: Prediction for the demand in [W]
        - supim_prediction_w: Prediction fot the PV generation of all three inverters
        - year: current year for non-intertemporal problems

    Returns:
        a dict of up to 12 DataFrames
    """


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

    # Convert 'kwp' to type float
    kwp = float(kwp)

    # Append the prediction to the according np.Arrays and remove the first element for this (length must remain)
    supim_w  = np.append(supim_w, supim_prediction_w)
    supim_w  = np.delete(supim_w, 0)
    demand_w = np.append(demand_w, demand_prediction_w)
    demand_w = np.delete(demand_w, 0)

    # Convert 'supim_w' and 'demand_w' from [W] to [kW]
    supim_kW = supim_w / 1e3 * supim_scale_factor
    demand_kW = demand_w / 1e3 * demand_scale_factor

    # Norm 'supim_kW' to peak power 'kwp'
    supim_norm = supim_kW / kwp

    # Convert 'supim_norm' and 'demand_kW' from array to list
    supim_norm = supim_norm.tolist()
    demand_kW  = demand_kW.tolist()

    # Check if len(demand) == len(supim)
    len_demand = len(demand_kW)
    len_supim  = len(supim_norm)
    if not len_demand == len_supim:
        raise Exception("PV supply and demand must have the same length!")


    # Generate global_prop df
    global_prop = {"Property":["CO2 limit", "Cost limit"],
                "value":[150000000.0, float("NaN")],
                "description": ["", ""]
                }

    global_prop = pd.DataFrame(global_prop)
    global_prop.set_index("Property", inplace=True)
    global_prop = pd.concat([global_prop], keys=[year],
                                        names=['support_timeframe'])
    gl.append(global_prop)


    # Generate site df
    site = {"Name":["coses"],
            "area":[int(kwp)]
            }

    site = pd.DataFrame(site)
    site.set_index("Name", inplace=True)
    site = pd.concat([site], keys=[year],
                                        names=['support_timeframe'])
    sit.append(site)


    # Generate commodity df
    commodity = {"Site":        ["coses","coses","coses","coses","coses","coses"], # different to df from Excel
                "Commodity":   ["Solar","Electricity","Slack","CO2","Elec-buy","Elec-sell"],
                "Type":        ["SupIm","Demand","Stock","Env","Buy","Sell"],
                "price":       [0.09,0.00,0.00,0.00,1.00,1.00],
                "max":         [kwp,float("inf"),float("inf"),float("inf"),float("inf"),float("inf")],
                "maxperhour":  [kwp,float("inf"),float("inf"),float("inf"),float("inf"),float("inf")]
            }

    commodity = pd.DataFrame(commodity)
    commodity.set_index(['Site', 'Commodity', 'Type'], inplace=True)
    commodity = pd.concat([commodity], keys=[year], names=['support_timeframe'])
    com.append(commodity)


    # Generate process df
    process =    {"Site":        ["coses","coses","coses","coses"],
                "Process":        ["Photovoltaics","Slack powerplant","Purchase","Feed-in"],
                "inst-cap":   [18,99,0,0],
                "cap-lo":        [18,99,0,0],
                "cap-up":       [18,99,10,10],
                "max-grad":         [float("inf"),float("inf"),float("inf"),float("inf")],
                "min-fraction":  [0,0,0,0],
                "inv-cost":  [0,0,0,0],
                "fix-cost":  [0,0,0,0],
                "var-cost":  [0,100,0,0],
                "startup-cost":  [0,0,0,0],
                "wacc":  [0.07,0.07,0.07,0.07],
                "depreciation":  [20,20,20,20],
                "area-per-cap":  [1,float("NaN"),float("NaN"),float("NaN")]
            }

    process = pd.DataFrame(process)
    process.set_index(['Site', 'Process'], inplace=True)
    process = pd.concat([process], keys=[year], names=['support_timeframe'])
    pro.append(process)


    # Generate process df
    process_commodity =    {"Process":        ["Photovoltaics","Photovoltaics","Slack powerplant","Slack powerplant","Purchase","Purchase","Purchase","Feed-in","Feed-in"],
                            "Commodity":      ["Solar","Electricity","Slack","Electricity","Elec-buy","Electricity","CO2","Electricity","Elec-sell"],
                            "Direction":      ["In","Out","In","Out","In","Out","Out","In","Out"],
                            "ratio":          [1.00,1.00,1.00,1.00,1.00,1.00,0.56,1.00,1.00],
                            "ratio-min":      [float("NaN"),float("NaN"),float("NaN"),float("NaN"),float("NaN"),float("NaN"),float("NaN"),float("NaN"),float("NaN")]
            }

    process_commodity = pd.DataFrame(process_commodity)
    process_commodity.set_index(['Process', 'Commodity', 'Direction'], inplace=True)
    process_commodity = pd.concat([process_commodity], keys=[year], names=['support_timeframe'])
    pro_com.append(process_commodity)



    # Generate process df
    demand =    {"t":                     [i for i in range(0,len_demand)],
                "coses.Electricity":      demand_kW
            }

    demand = pd.DataFrame(demand)
    demand.set_index('t', inplace=True)
    demand = pd.concat([demand], keys=[year], names=['support_timeframe'])
    demand.columns = split_columns(demand.columns, '.')
    dem.append(demand)


    # Generate process df
    supim =    {"t":                [i for i in range(0,len_supim)],
                "coses.Solar":      supim_norm
            }

    supim = pd.DataFrame(supim)
    supim.set_index('t', inplace=True)
    supim = pd.concat([supim], keys=[year], names=['support_timeframe'])
    supim.columns = split_columns(supim.columns, '.')
    sup.append(supim)


    # Generate storage df
    # 2 BSS
    # storage = { "Site":         ["coses"],
    #             "Storage":      ["Battery"],
    #             "Commodity":    ["Electricity"],
    #             "inst-cap-c":   [12], # 12
    #             "cap-lo-c":     [0], # 12
    #             "cap-up-c":     [12], # 12
    #             "inst-cap-p":   [12], # 12
    #             "cap-lo-p":     [0], # 12
    #             "cap-up-p":     [12], # 12
    #             "eff-in":       [0.97],
    #             "eff-out":      [0.97],
    #             "inv-cost-p":   [0],
    #             "inv-cost-c":   [0],
    #             "fix-cost-p":   [0],
    #             "fix-cost-c":   [0],
    #             "var-cost-p":   [0],
    #             "var-cost-c":   [0],
    #             "depreciation": [20],
    #             "wacc":         [0.07],
    #             "init":         [soc_init],
    #             "discharge":    [0.000003],
    #             "ep-ratio":     [1]
    #         }
    
    # 1 BSS
    storage_capacity_kWh = 6
    storage_power_kW = 6
    storage = { "Site":         ["coses"],
                "Storage":      ["Battery"],
                "Commodity":    ["Electricity"],
                "inst-cap-c":   [storage_capacity_kWh], # 12
                "cap-lo-c":     [storage_capacity_kWh], # 12
                "cap-up-c":     [storage_capacity_kWh], # 12
                "inst-cap-p":   [storage_power_kW], # 12
                "cap-lo-p":     [storage_power_kW], # 12
                "cap-up-p":     [storage_power_kW], # 12
                "eff-in":       [0.97],
                "eff-out":      [0.97],
                "inv-cost-p":   [0],
                "inv-cost-c":   [0],
                "fix-cost-p":   [0],
                "fix-cost-c":   [0],
                "var-cost-p":   [0],
                "var-cost-c":   [0],
                "depreciation": [20],
                "wacc":         [0.07],
                "init":         [soc_init], #soc_init
                "discharge":    [0.01],
                "ep-ratio":     [1]
            }

    storage = pd.DataFrame(storage) # 'storage' is provided by function argument by the Urbs Wrapper
    storage.set_index(['Site', 'Storage', 'Commodity'], inplace=True)
    storage = pd.concat([storage], keys=[year],
                                        names=['support_timeframe'])
    sto.append(storage)


    # Generate buySellPrices df
    buy_sell_price =    {"t":          [i for i in range(0,len_demand)],
                        "Elec-buy":    [price_elec_buy  for i in range(0,len_demand)],
                        "Elec-sell":   [price_elec_sell for i in range(0,len_demand)]
            }

    buy_sell_price = pd.DataFrame(buy_sell_price)
    buy_sell_price.set_index("t", inplace=True)
    buy_sell_price = pd.concat([buy_sell_price], keys=[year],
                                        names=['support_timeframe'])
    buy_sell_price.columns = split_columns(buy_sell_price.columns, '.')
    bsp.append(buy_sell_price)

    # Generate empty df's for transmission and eff
    transmission = pd.DataFrame()
    tra.append(transmission)
    dsm = pd.DataFrame()
    ds.append(dsm)
    eff_factor = pd.DataFrame()
    ef.append(eff_factor)

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


    return (data)