import os
import pyomo.environ
from pyomo.opt.base import SolverFactory
from datetime import datetime, date
from .model import create_model
from .report import *
from .plot import *
from .input import *
from .input_direct import read_input_direct
from .validation import *
from .saveload import *
from .iot_server_data import *
from telemetry import tprint # type: ignore


def prepare_result_directory(result_name):
    """ create a time stamped directory within the result folder.

    Args:
        result_name: user specified result name

    Returns:
        a subfolder in the result folder 
    
    """
    # timestamp for result directory
    now = datetime.now().strftime('%Y%m%dT%H%M')

    # create result directory if not existent
    result_dir = os.path.join('output-urbs-mpc', '{}-{}'.format(result_name, now))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir


def setup_solver(optim, logfile='solver.log'):
    """ """
    if optim.name == 'gurobi':
        # reference with list of option names
        # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
        optim.set_options("logfile={}".format(logfile))
        # optim.set_options("timelimit=7200")  # seconds
        # optim.set_options("mipgap=5e-4")  # default = 1e-4
    elif optim.name == 'glpk':
        # reference with list of options
        # execute 'glpsol --help'
        optim.set_options("log={}".format(logfile))
        # optim.set_options("tmlim=7200")  # seconds
        # optim.set_options("mipgap=.0005")
    elif optim.name == 'cplex':
        optim.set_options("log={}".format(logfile))
    else:
        print("[ URBS ] Warning from setup_solver: no options set for solver "
              "'{}'!".format(optim.name))
    return optim


def run_scenario(input_files, Solver, timesteps, scenario, result_dir, dt,
                 objective, plot_tuples=None,  plot_sites_name=None,
                 plot_periods=None, report_tuples=None,
                 report_sites_name=None,
                 iot_server=None):
    """ run an urbs model for given input, time steps and scenario

    Args:
        - input_files: filenames of input Excel spreadsheets
        - Solver: the user specified solver
        - timesteps: a list of timesteps, e.g. range(0,8761)
        - scenario: a scenario function that modifies the input data dict
        - result_dir: directory name for result spreadsheet and plots
        - dt: length of each time step (unit: hours)
        - objective: objective function chosen (either "cost" or "CO2")
        - plot_tuples: (optional) list of plot tuples (c.f. urbs.result_figures)
        - plot_sites_name: (optional) dict of names for sites in plot_tuples
        - plot_periods: (optional) dict of plot periods
          (c.f. urbs.result_figures)
        - report_tuples: (optional) list of (sit, com) tuples
          (c.f. urbs.report)
        - report_sites_name: (optional) dict of names for sites in
          report_tuples

    Returns:
        the urbs model instance
    """

    # sets a modeled year for non-intertemporal problems
    # (necessary for consitency)
    year = date.today().year

    # scenario name, read and modify data for scenario
    sce = scenario.__name__

    # data = read_input(input_files, year)
    data = read_input_direct(2024)
    # Wrap the IoT Server data to the required dict()
    # data = read_input_from_iot_server(iot_server)
    
    data = scenario(data)
    validate_input(data)
    validate_dc_objective(data, objective)

    # create model
    prob = create_model(data, dt, timesteps, objective)
    # prob = create_model(data=data, objective=objective) # self-made
    # prob_filename = os.path.join(result_dir, 'model.lp')
    # prob.write(prob_filename, io_options={'symbolic_solver_labels':True})

    # refresh time stamp string and create filename for logfile
    log_filename = os.path.join(result_dir, '{}.log').format(sce)

    # solve model and read results
    # optim = SolverFactory(Solver)  # cplex, glpk, gurobi, ...
    optim = SolverFactory(Solver, executable='C:\\glpk-4.65\\w64\\glpsol')  # changed
    optim = setup_solver(optim) #, logfile=log_filename)
    result = optim.solve(prob, tee=True)
    assert str(result.solver.termination_condition) == 'optimal'

    # save problem solution (and input data) to HDF5 file
    save(prob, os.path.join(result_dir, '{}.h5'.format(sce)))

    # write report to spreadsheet
    report(
        prob,
        os.path.join(result_dir, '{}.xlsx').format(sce),
        report_tuples=report_tuples,
        report_sites_name=report_sites_name)

    # result plots
    result_figures(
        prob,
        os.path.join(result_dir, '{}'.format(sce)),
        timesteps,
        plot_title_prefix=sce.replace('_', ' '),
        plot_tuples=plot_tuples,
        plot_sites_name=plot_sites_name,
        periods=plot_periods,
        figure_size=(24, 9))

    return prob


def run_scenario_online(timesteps, 
                        scenario, 
                        result_dir, 
                        dt,
                        objective="cost",
                        kwp:float=18.0,
                        demand_prediction_w:float=0.0,
                        supim_prediction_w:float=0.0,
                        demand_scale_factor:float=1.0,
                        supim_scale_factor:float=1.0,
                        report_tuples=None,
                        report_sites_name=None,
                        df=pd.DataFrame(),
                        solver="glpk",
                        solverpath="",
                        soc0_last_state=0.0,
                        soc1_last_state=0.0,
                        verbose=False,
                        ):
    """ run an urbs model for given input, time steps and scenario
        Input is based on data from the IoT-Server of the CoSES

    Args:
        - Solver: the user specified solver
        - timesteps: a list of timesteps, e.g. range(0,8761)
        - scenario: a scenario function that modifies the input data dict
        - result_dir: directory name for result spreadsheet and plots
        - dt: length of each time step (unit: hours)
        - objective: objective function chosen (either "cost" or "CO2")
          (c.f. urbs.result_figures)
        - report_tuples: (optional) list of (sit, com) tuples
          (c.f. urbs.report)
        - report_sites_name: (optional) dict of names for sites in
          report_tuples
        - df: pd.DataFrame from the IoT Server with the corresponding measurements

    Returns:
        the urbs model instance
    """

    # sets a modeled year for non-intertemporal problems
    # (necessary for consitency)
    year = date.today().year

    # scenario name, read and modify data for scenario
    sce = scenario.__name__

    # Approximate the power generation of Inverter 1 (values from Dez/05 2025, 11:45 am)
    P_inv1_static = 3867 # [w]
    P_inv2_static = 2038 # [w]
    P_inv3_static = 1966 # [w]
    factor_inv1   = (P_inv1_static) / (P_inv2_static + P_inv3_static)

    # Read Input from df by the IoT-Server
    inv2 = df.INV2.values
    inv3 = df.INV3.values
    inv1 = (inv2 + inv3) * factor_inv1
    inverter_w = (inv1 + inv2 + inv3)
    pm_server = df.SHELLY_API_SERVERROOM_POWER.values
    pm_office = df.SHELLY_API_FLOOR_POWER.values
    demand_w = (pm_server + pm_office)

    # Process the BSS output power (Charge --> Demand, Discharge --> SupIm)
    try:
        e3dc0power_ac = df.E3DC0POWER_AC0.values + df.E3DC0POWER_AC1.values + df.E3DC0POWER_AC2.values
        for idx, power in enumerate(e3dc0power_ac):
            if power > 0: # Charge --> Demand [W]
                demand_w[idx] += power
            elif power < 0: # Discharge --> 'Inverter' [W]
                inverter_w[idx] += power
            pass
    except Exception as e:
        pass

    try:
        e3dc1power_ac = df.E3DC1POWER_AC0.values + df.E3DC1POWER_AC1.values + df.E3DC1POWER_AC2.values
        for idx, power in enumerate(e3dc1power_ac):
            if power > 0: # Charge --> Demand [W]
                demand_w[idx] += power
            elif power < 0: # Discharge --> 'Inverter' [W]
                inverter_w[idx] += power
            pass
    except Exception as e:
        pass



    
    try:
        soc0 = df.E3DC0SOC.values
        # soc0 = df["E3DC0SOC"].tolist() 
        soc0_first = float(soc0[0]) / 100
        # soc0_last = float(soc0[-1]) / 100
        soc0_last = soc0_last_state # directly from IoT Server with res=1m 
        tprint(f"SoC[ 0]: {(soc0_first*100):.2f} %", "URBS") 
        tprint(f"SoC[-1]: {(soc0_last *100):.2f} %", "URBS")
    except Exception as e:
        tprint(e, "URBS")
        soc0_first = 0
        soc0_last  = 0
        
    try:
        soc1 = df.E3DC1SOC.values
        soc1_first = float(soc1[0]) / 100
        # soc1_last  = float(soc1[-1]) / 100
        soc1_last = soc1_last_state # directly from IoT Server with res=1m 
    except:
        soc1_first = 0
        soc1_last  = 0


    # Since Urbs wants to reach the Soc_init value within the last step, soc_last is set as init
    data = read_input_direct(supim_w=inverter_w,
                             demand_w=demand_w,
                            #  soc_init=soc0_first+soc1_first,
                             soc_init=soc0_last+soc1_last,
                             price_elec_buy=0.35,
                             price_elec_sell=0.0,
                             demand_prediction_w=demand_prediction_w,
                             supim_prediction_w=supim_prediction_w,
                             demand_scale_factor=demand_scale_factor,
                             supim_scale_factor=supim_scale_factor,
                             kwp=kwp, 
                             year=year)


    
    data = scenario(data)
    validate_input(data)
    validate_dc_objective(data, objective)

    # Create model
    # timesteps = None
    prob = create_model(data, 
                        dt, 
                        timesteps, 
                        objective, 
                        hours=192, # 8760
                        verbose=verbose)

    # Define the solver to be used
    if solverpath == "":
        # '' (empty str) is set under unix environments, where e.g glpk is available in apt-packages
        optim = SolverFactory(solver)
    else:
        # A direct path is set under environment like windows, where the solver needs to be downloaded manually
        optim = SolverFactory(solver, executable=solverpath)

    optim = setup_solver(optim) #, logfile=log_filename)

    result = optim.solve(prob, tee=verbose) # 'tee' enables the verbose mode for the solver
    if verbose:
        tprint(result, "URBS")

    assert str(result.solver.termination_condition) == 'optimal'

    # save problem solution (and input data) to HDF5 file
    # save(prob, os.path.join(result_dir, '{}.h5'.format(sce)))

    # write report to spreadsheet and get the results as a dict
    results = report(
                prob,
                os.path.join(result_dir, '{}.xlsx').format(sce),
                report_tuples=report_tuples,
                report_sites_name=report_sites_name)

    return prob, results

