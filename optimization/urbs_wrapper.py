"""
Urbs wrapper to run the optimization directly from an other 
script / notebook, etc with live data provided by the IoT Server

Author:     Benedikt Fuchsgruber
Mail:       benedikt.fuchsgruber@tum.de
Project:    Master's Thesis (MPC for the ZEI DERs)

"""

# Imports
import urbs
import sys
from typing import Literal, List
import glob
import pandas as pd
import numpy as np
import os
import math
from toolkit.telemetry import tprint # type: ignore

# Define function to extract the sign of a signed-float/int
sign = lambda x: math.copysign(1, x)


class UrbsWrapper():
    """
    Wrapper from data, derived by the IoT-Server to 'Urbs'-like data formats
    !! Work consistently with the named units !!

    Args:
        - _solverpath       Direct / absolute path to the solver's folder
        - _solver           Name of the solver
        - _objective        cost, CO2
        - _pv_max_Wp        Max installed PV power in Wp
        - _bss_max_Wh       Max capacity of the installed storages in Wh
        - _bss_strategy     Define how to control the BSS
        - _df               pd.DataFrame from the IoT-Server with the related measurements
        - _verbose          Activate serial logging
        - _iot_server_soc0  IoT-Server object for deriving the SoC of the BSS0
        - _iot_server_soc1  IoT-Server object for deriving the SoC of the BSS1

        
    Sequence:
        - Create object b< calling the constructor
        - set_df()
        - set_prediction_*()
        - run()
        - get_control_target()

    """

    def __init__(self, 
                 _solverpath:str="",
                 _solver:str=Literal["glpk", "gurobi", "cplex"],
                 _objective:str=Literal["cost", "CO2"],
                 _pv_max_Wp:float=18*1e3,
                 _bss_max_Wh:float=6*1e3,
                 _bss_strategy:str=Literal["equal", "successively", "one-only"],
                 _df:pd.DataFrame=pd.DataFrame(),
                 _verbose:bool=False,
                 _iot_server_soc0:pd.DataFrame=pd.DataFrame(),
                 _iot_server_soc1:pd.DataFrame=pd.DataFrame()
                 ) -> None :
        
        # Wrap constructor arguments into attributes
        self.solverpath      = f"{_solverpath}"
        self.solver          = _solver
        self.objective       = _objective
        self.data_df         = _df
        self.pv_max_kWp      = _pv_max_Wp / 1e3 # [Wp] to the common unit [kWp]
        self.bss_strategy    = _bss_strategy
        self.bss_max         = _bss_max_Wh  # input in [W] 
        self.bss_max_each    = self.bss_max / 2
        self.verbose         = _verbose
        self.iot_server_soc0 = _iot_server_soc0
        self.iot_server_soc1 = _iot_server_soc1

        # Define attributes for the prediction
        self.demand_prediction_w = 0.0
        self.supim_prediction_w  = 0.0

        # Define attributes for the last values of Demand and SupIm
        self.demand_last_w = 0.0
        self.supim_last_w  = 0.0

        # Define other attributes which are necessary for the optimization
        self.__doc__                = "Wrapper for IoT-Server measurement data to an 'Urbs'-like data format" 
        self.timesteps              = range(0, 24*2*4) # = 192
        self.dt                     = 0.25 # [hours]
        self.scenario               = urbs.scenario_base 
        self.report_tuples          = [ (2024, 'coses', 'Electricity') ]
        self.report_sites_name      = {"coses": "coses"}
        self.input_dir              = "Input/direct"
        self.cwd                    = os.getcwd()

        # Define the dictionary for the input data
        self.results            = {} # [dict]
        self.prob               = None
        self.optimized          = False # [bool]
        self.epr                = 1 # [hours]
        self.bss                = {} # Control strategy for the two BSS 
        self.bss0_lastSoC       = 0.0 # [0 --> 1]
        self.bss1_lastSoC       = 0.0 # [0 --> 1]
        self.soc_target         = 0.0 # [0 --> 1]
        self.dSoc               = 0.0 # [kWh]
        self.soc_limit_low      = 0.05 # [0 --> 1]
        self.target_charge      = 0.0 # [W]
        self.target_discharge   = 0.0 # [W]
        self.target             = 0.0 # Calculated with 'target_charge' and 'target_discharge'
        self.demand_scale_factor = 1.0
        self.supim_scale_factor = 1.0
        self.soc_init           = 0.0

        # Simulation flag as float
        self.simulation = 0.0


        tprint(f"UrbsWrapper with solver '{self.solver}' for objective '{self.objective}' initialized", "URBS")
        pass


    def __str__(self):
        """
        Implementation of the Python print method
        """
        return "Urbs wrapper"
    

    def prepare_result_dir(self):
        """
        Prepare the output folder for optimization stuff like the Urbs framework does
        """
        result_name = 'urbs-mpc'
        result_dir = urbs.prepare_result_directory(result_name)  # name + time stamp
        # copy input file to result directory
        try:
            # shutil.copytree(input_path, os.path.join(result_dir, input_dir))
            pass # TODO Copy the downloaded df's there
        except NotADirectoryError:
            # shutil.copyfile(input_path, os.path.join(result_dir, input_files))
            pass # TODO Copy the downloaded df's there

        return result_dir
    
    
    def calc_last_soc(self):
        """
        Derive the values for the last SoC entry from the df in the range [0...1]
        Idea is that the SoC should not be middled over the 15min time range.
        However, own IoT-Server objects with res=1m are used here
        --> Called by self.set_df() and so by the MPC cycle
        """
        try: # for bss0
            # self.bss0_lastSoC = self.data_df.E3DC0SOC.values[-1] / 100
            self.iot_server_soc0.get_df()
            self.bss0_lastSoC = self.iot_server_soc0.getLastValue() / 100
            pass
        except:
            self.bss0_lastSoC = 0.0

        try: # for bss1
            # self.bss1_lastSoC = self.data_df.E3DC1SOC.values[-1] / 100
            self.iot_server_soc1.get_df()
            self.bss1_lastSoC = self.iot_server_soc1.getLastValue() / 100
            pass
        except:
            self.bss1_lastSoC = 0.0
    

    def set_prediction_demand(self, pred) -> None:
        """
        Set the prediction value for the demand in [W]
        --> Called by MPC cyle
        """
        self.demand_prediction_w = pred
        return None


    def set_prediction_pv(self, pred) -> None:
        """
        Set the prediction value for the pv in [W] for all three inverters
        --> Called by MPC cyle
        """
        # Limit the PV prediction to 0
        if pred <= 0:
            pred = 0.1 # very small value necessary in order to prevent Urbs from ignoring the PV

        self.supim_prediction_w = pred
        return None

    
    def set_df(self, df) -> None:
        """
        Set df with a object with type pd.DataFrame
        --> Called by MPC cyle
        """
        self.data_df = df
        self.calc_last_soc()
        self.demand_last_w = self.data_df.SHELLY_API_SERVERROOM_POWER.values[-1] + self.data_df.SHELLY_API_FLOOR_POWER.values[-1]
        self.supim_last_w  = (self.data_df.INV2.values[-1] + self.data_df.INV3.values[-1])/2*3


    def set_supIm_scale(self, scale) -> None:
        """
        Set a scaling factor in order to influence the PV Generation
        --> Can be called once or cyclic
        """
        self.supim_scale_factor = scale


    def set_demand_scale(self, scale) -> None:
        """
        Set a scaling factor in order to influence the demand side
        --> Can be called once or cyclic
        """
        self.demand_scale_factor = scale


    def run(self) -> None:
        """
        Start to solve the optimization problem with Urbs
        --> Called by MPC cycle
        """

        # Call a modified Urbs-method to run the optimization
        # --> Results are stored within the class object

        if self.simulation == 1.0:
            tprint("BSS Simulation is active", "URBS")

        self.prob, self.results = urbs.run_scenario_online( timesteps=self.timesteps,
                                                            scenario=self.scenario,
                                                            result_dir=self.prepare_result_dir(),
                                                            dt=self.dt,
                                                            objective=self.objective,
                                                            kwp=self.pv_max_kWp*self.supim_scale_factor,
                                                            demand_prediction_w=self.demand_prediction_w,
                                                            supim_prediction_w=self.supim_prediction_w,
                                                            demand_scale_factor=self.demand_scale_factor,
                                                            supim_scale_factor=self.supim_scale_factor,
                                                            report_tuples=self.report_tuples,
                                                            report_sites_name=self.report_sites_name,
                                                            df=self.data_df,
                                                            solver=self.solver,
                                                            solverpath=self.solverpath,
                                                            soc0_last_state=self.bss0_lastSoC,
                                                            soc1_last_state=self.bss1_lastSoC,
                                                            verbose=self.verbose,
                                                           )
        self.optimized = True
        pass


    def set_bss_max(self, max_p) -> None:
        """
        Set the ouput power maximum for one BSS in [W].
        Normally in range (0,6000)
        """
        self.bss_max = max_p
        return None


    def get_control_target(self, weight_with_dt:bool=True) -> dict:
        """
        Get the control target from the Urbs optimization
        Please note:
            Since Urbs is not weighting dt correctly and so the bss power is too small by the factor of dt,
            this can be calculated here.
        
        Returns:
            - Signed control target for each of the both BSS
            Positive: Charge the related BSS
            Negative: Discharge the related BSS and feed into the microgrid
        """

        self.bss = {
                    "bss_number":"signed_target_in_W", # __doc__,
                    "simulation":self.simulation,
                    "strategy":self.bss_strategy,
                    0: 0.0,
                    1: 0.0
                    }
        
        target_bss_0 = 0.0
        target_bss_1 = 0.0

        if not self.optimized:
            self.run()

        # Get the storage-related values from the optimization
        strategy_storage = self.results["storage"]
        level = strategy_storage.Level.values # predicted storage SoC from urbs optimization
        charge = strategy_storage.Stored.values * 1000 # [W]
        discharge = strategy_storage.Retrieved.values * 1000 # [W]

        # Calculate the power-target with help of the target-SoC for the BSS
        self.soc_target = (level[-2]) / (self.bss_max/1000) # returns the target-SoC in [0 --> 1]
        # self.soc_actual = (self.data_df.E3DC0SOC.values[-1]) / 100 # [0 --> 1]
        self.soc_actual = self.bss0_lastSoC

        # Check, if the battery is to empty
        if (self.soc_actual <= self.soc_limit_low) or (self.soc_target < self.soc_limit_low):
            self.soc_target = self.soc_limit_low

        self.dSoc = self.soc_target - self.soc_actual
        self.dEnergy = self.bss_max * self.dSoc
        self.target = self.dEnergy / self.epr #/ self.dt  # [W]


        # Log the related SoC values to the console
        tprint(f"SoC_target: {(self.soc_target*100):.2f} %", "URBS")
        tprint(f"dSoC: {(self.dSoc*100):.2f} %", "URBS")



        # # Get the target (= last values)
        # self.target_charge = charge[-1]
        # self.target_discharge = discharge[-1]

        # # Define the energy flow direction
        # self.target = 0.0 # Reset
        # if (self.target_charge > self.target_discharge):
        #     self.target = self.target_charge
        # else:
        #     self.target = - self.target_discharge




        # Decide which BSS should by used first
        if self.bss_strategy == "successively":
            # Use one BSS first, until the target power is greater than the limit
            if abs(self.target) <= self.bss_max_each:
                target_bss_0 = self.target
                target_bss_1 = 0.0
            else:
                target_bss_0 = self.bss_max_each * sign(self.target)        # BSS_0 is completely utilizied
                target_bss_1 = self.target - target_bss_0                   # BSS_1 does the missing part
            pass
        elif self.bss_strategy == "equal":
            # Devide the target power equaly between the two BSS
            target_bss_0 = self.target / 2
            target_bss_1 = target_bss_0
        elif self.bss_strategy == "one-only":
            # Use only the first BSS --> second one is always 0 W
            target_bss_0 = self.target
            target_bss_1 = 0.0

            pass
        
        if weight_with_dt:
            self.bss[0] = target_bss_0 / self.dt
            self.bss[1] = target_bss_1 / self.dt
        else:
            self.bss[0] = target_bss_0
            self.bss[1] = target_bss_1
        pass
        return self.bss
    

    def activate_simulation(self) -> None:
        self.simulation = 1.0

    
    def deactivate_simulation(self) -> None:
        self.simulation = 0.0
        

    def get_results(self) -> pd.DataFrame:
        """
        Get the results of the last MPC cycle
        --> Called by the MPC cycle and given to the Evaluation object

        Returns:
            - pd.DataFrame with different cols and one value each
        """

        self.mpc_results = {
            "power_target0":[self.bss[0]],
            "power_target1":[self.bss[1]],
            "soc0":[(self.bss0_lastSoC)*100],
            "soc1":[(self.bss1_lastSoC)*100],
            "soc_target":[(self.soc_target)*100],
            "dsoc":[(self.dSoc)*100],
            "prediction_demand":[self.demand_prediction_w],
            "prediction_supim":[self.supim_prediction_w],
            "demand":[self.demand_last_w],
            "supim":[self.supim_last_w],
            "demand_scale_factor":[self.demand_scale_factor],
            "supim_scale_factor":[self.supim_scale_factor],
            "simulation": [self.simulation]
        }
        
        self.mpc_df = pd.DataFrame(self.mpc_results)
        pass
        return self.mpc_df


