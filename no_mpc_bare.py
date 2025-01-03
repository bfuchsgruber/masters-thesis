# Imports
import sys
sys.path.append('toolkit')
sys.path.append('prediction')
sys.path.append('optimization')
sys.path.append('control')
sys.path.append('evaluation')
print(sys.path)

from imports import * # type: ignore
from iot_grabber import IotGrabber # type: ignore
from toolkit.prediction import Prediction # type: ignore
from urbs_wrapper import UrbsWrapper # type: ignore
from control import control_bss # type: ignore
from evaluation import Evaluation # type: ignore

import pandas as pd
import numpy as np
import time
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


# IoT Server
time_range  = "48h"
ip          = "100.113.141.113"

iot_server = IotGrabber(ip=ip, range=time_range)
iot_server.activate(inverter   = True) # Append the inverter
iot_server.activate(powermeter = True) # Append the Shelly power measurements
iot_server.activate(bss0       = True) # Append the E3DC 0

# Prediction
prediction_models = {"INV1": f"prediction/lstm/final/LSTM-Inverter1.model",
                     "INV2": f"prediction/lstm/final/LSTM-Inverter2.model",
                     "INV3": f"prediction/lstm/final/LSTM-Inverter3.model",
                     "SHELLY_API_SERVERROOM_POWER": f"prediction/lstm/final/LSTM-Serverroom.model",
                     "SHELLY_API_FLOOR_POWER": f"prediction/lstm/final/LSTM-Floor.model"
                     }

# Optimization
solverpath = "optimization/glpk-4.65"
urbs = UrbsWrapper(_solverpath=solverpath,
                   _solver="glpk",
                   _objective="cost",
                   _pv_max_Wp=18.0 * 1e3,
                   _bss_max_W=6000,
                   _bss_strategy="one-only",
                #    _bss_strategy="successively",
                    # _bss_strategy="equal",
                   )
# urbs.set_supIm_scale(4)

# Controller
control_bss0 = control_bss.Control(bss_index=0,
                                   mqtt_broker="100.113.141.113")

# Evalutation
evaluation = Evaluation(mqtt_broker=iot_server.getIp())


# MPC Algorithm without MPC --> only Urbs

# delay = 15*60 # 15 min
delay = 1*60 # 1 min
duration_counter = 0
time_last_duration = time.time() - (1 * delay)

while True:
    # Calculate the current time and delta to last cycle
    time_actual = time.time()
    dT = (time_actual - time_last_duration)

    # MPC
    if dT >= delay:
        print(f"[ MPC ] MPC algorithm startet, cylce: {duration_counter}")
        print(f"[ MPC ] dT: {dT} sec")

        # Download new values
        iot_server.setRange("48h")
        iot_server.setRes("1m")
        df = iot_server.get_df()
        

        # Make a new forecast for the PV and the loads
        model_inv2 = Prediction(prediction_models["INV2"],
                                    "INV2",
                                    previous_df=df,
                                    verbose=False)
        model_inv3 = Prediction(prediction_models["INV3"],
                                    "INV3",
                                    previous_df=df,
                                    verbose=False)
        model_serverroom = Prediction(prediction_models["SHELLY_API_SERVERROOM_POWER"],
                                    "SHELLY_API_SERVERROOM_POWER",
                                    previous_df=df,
                                    verbose=False)
        model_floor = Prediction(prediction_models["SHELLY_API_FLOOR_POWER"],
                                    "SHELLY_API_FLOOR_POWER",
                                    previous_df=df,
                                    verbose=False)
        
        # forecast_inv2 = model_inv2.predict(convert_utc_to_local=True, float_output=True)
        # forecast_inv3 = model_inv3.predict(convert_utc_to_local=True, float_output=True)
        # forecast_serverroom = model_serverroom.predict(convert_utc_to_local=True, float_output=True)
        # forecast_floor = model_floor.predict(convert_utc_to_local=True, float_output=True)

        # Set the las value of the column as prediction --> no change so far!
        forecast_inv2 = float(df.INV2.values[-1])
        forecast_inv3 = float(df.INV3.values[-1])
        forecast_serverroom = float(df.SHELLY_API_SERVERROOM_POWER.values[-1])
        forecast_floor = float(df.SHELLY_API_FLOOR_POWER.values[-1])


        # Extrapolate the PV yield for Inverter 1
        prediction_inv = (forecast_inv2 + forecast_inv3)/2*3
        prediction_demand = forecast_serverroom + forecast_floor


        # Run Optimization
        urbs.set_df(df)
        urbs.set_prediction_demand(prediction_demand)
        urbs.set_prediction_pv(prediction_inv)
        # urbs.set_supIm_scale(15)
        urbs.run()
        bss = urbs.get_control_target(weight_with_dt=False)
        bss_target0 = bss[0]
        bss_target1 = bss[1]
        print(bss_target0)
        # control_bss0.direct_control(bss_target0) # no output!


        # Evaluate the MPC cycle
        mpc_results = urbs.get_results()
        evaluation.cycle_without_mpc(mpc_results)

        # Prepare duration end of this cycle
        pass
        # break # DEBUG
        duration_counter += 1
        time_last_duration = time.time()
        print(f"[ MPC ] Duration time: {time_last_duration-time_actual} sec")
        print(f"------------------------------------------------")

    # Short cycle delay in [sec]
    time.sleep(0.01) 



# END OF MPC