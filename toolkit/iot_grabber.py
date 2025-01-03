"""
Class for grabbing measurement data sets automatically from the IoT-Server's backend service at port 5000

Author:     Benedikt Fuchsgruber
Mail:       benedikt.fuchsgruber@tum.de
Project:    Master's Thesis (MPC for the ZEI DERs)

"""

# Imports
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
import os
from prettytable import PrettyTable
import re
from matplotlib import pyplot as plt
from telemetry import tprint


# Class Definition
class IotGrabber():
    """
    Class for the IoT-Server Grabber

    Usage:
        - Create an instance of this class and set all values according to the constructor
        - (Everything can be set after that again with the Setter-Methods)
        - Decide whether an absolute time or a relative range from now() is required
        - call 'get_df' to request measurements and save them as CSV / pd.Dataframe
    """

    def __init__(self, 
                 ip:str="100.113.141.113", 
                 devices:list=[],
                 time_abs_start:str="", #"2024-09-09T11:00:00Z",
                 time_abs_end  :str="",#"2024-09-11T11:00:00Z",
                 range:str="None",
                 res:str="15m",
                 prediction_horizont:int=15,
                 prediction_horizont_unit:str="min",
                 delimiter=";",
                 cwd:str=os.getcwd(),
                 verbose:bool=False 
                ) -> None:
        
        # Set all input parameters
        self.setIp(ip=ip)
        self.setDevices(device_list=devices)
        self.setTimeAbsStart(time_abs_start=time_abs_start)
        self.setTimeAbsEnd(time_abs_end=time_abs_end)
        self.setRange(range=range)
        self.setRes(res=res)
        self.setPredictionHorizont(horiz=prediction_horizont)
        self.setPredictionHorizontUnit(unit=prediction_horizont_unit)
        self.setDelimiter(delimiter=delimiter)
        self.setCwd(cwd=cwd)
        self.verbose = verbose

        # Initialize an empty pd.Dataframe
        self.got_df = pd.DataFrame()
        self.df_mpc = pd.DataFrame()

        # Check if IoT Server is reachable
        self.ping = False
        self.check_ping()

        # Calc prediction end time
        # self.calcPredictionEndTime()

        # Name of the index column in the InfluxDB derive
        self.cTime = "time"

        # Set Port to Flask Backend
        self.backend_port = 5000

        # Set Path to Backend
        self.backend_path = "download-file"

        # Set doc-string for help()-command
        self.__doc__ =     """
    Class for the IoT-Server Grabber

    Usage:
        - Create an instance of this class and set all values according to the constructor
        - (Everything can be set after that again with the Setter-Methods)
        - Decide whether an absolute time or a relative range from now() is required
        - call 'get_df' to request measurements and save them as CSV / pd.Dataframe
    """
        if self.verbose:
            if self.ping:
                tprint(f"IoT Server at {self.ip} initialized", "IOT")
            else:
                tprint(f"IoT Server at {self.ip} failed to initialize", "IOT")
            pass

    def check_ping(self) -> None:
        try:
            response = os.system("ping -n 1 " + self.ip)
            # and then check the response...
            if response == 0:
                self.ping = True
            else:
                self.ping = False
        except:
            self.ping = False
        
        # self.ping = True # Activate only for running the script on OS != Windows
        
    def __str__ (self) -> str:
        table = PrettyTable(["Field", "Value"])
        table.add_row([f"IP", f"{self.ip}"])
        table.add_row([f"Reachable", f"{self.ping}"])
        table.add_row([f"Devices", f"{self.devices}"])
        table.add_row([f"Time (abs)", f"{self.time_abs_start} < t < {self.time_abs_end}"])
        table.add_row([f"Time (rel)", f"{self.range}"])
        table.add_row([f"Time (res)", f"{self.res}"])
        table.add_row([f"Prediction horizont", f"{self.prediction_horizont}{self.prediction_horizont_unit}"])
        table.add_row([f"Delimiter", f"{self.delimiter}"])
        table.add_row([f"CWD", f"{self.cwd}"])
        table.add_row([f"Backend Port", f"{self.backend_port}"])
        table.add_row([f"Backend Path", f"{self.backend_path}"])

        return str(table)

    def setCwd(self, cwd) -> None:
        """
        Change the current working directory
        --> usefull if the script is a different directory than the master file
        """
        self.cwd = cwd

    def getCwd(self) -> str:
        return self.cwd
    
    def calcPredictionEndTime(self) -> None:
        time_end = datetime.strptime(self.time_abs_end, "%Y-%m-%dT%H:%M:%SZ")
        dT = timedelta(minutes=0)

        # match self.prediction_horizont_unit:
        #     case "min":
        #         dT = timedelta(minutes=self.prediction_horizont)
        #     case "hour":
        #         dT = timedelta(hours=self.prediction_horizont)
        #     case "sec":
        #         dT = timedelta(seconds=self.prediction_horizont)

        # self.time_abs_end_prediction = time_end + dT
        # self.time_abs_end_prediction = self.time_abs_end_prediction.strftime("%Y-%m-%dT%H:%M:%SZ")
        
    def get_df(self, norm=False) -> pd.DataFrame:
        """
        Request a pd.Dataframe out of the set measurements
        --> Returns the dataframe
        --> Saves everything as a CSV-file in the output directory
        """

        # Set cwd to self.cwd
        if self.verbose:
            tprint(f"Current directory before change: {os.getcwd()}", "IOT")
        os.chdir(self.cwd)
        if self.verbose:
            tprint(f"Current directory after change:  {os.getcwd()}", "IOT")

        # Check if output folder exists
        Path("output-iot-server").mkdir(parents=True, exist_ok=True)


        # Calc a new predition horizont
        # self.calcPredictionEndTime()

        # Check if connection to IoT Server is valid
        self.check_ping()

        if self.ping:
            # Build URL
            url = f"http://{self.ip}:{self.backend_port}/{self.backend_path}"

            # Init a Pandas DataFrame
            df = pd.DataFrame()
            df_prediction_horizont = pd.DataFrame()

            for idx, device in enumerate(self.devices):
                filename = f"{device}.csv"
                data = dict()
                data = {
                    "delimiter": self.delimiter,
                    "request": device,
                    "filename": filename,
                    "range": self.range,
                    "res": self.res,
                    "ip": self.ip
                }

                if self.range == "None":
                    data["time_abs"] = f"time >= '{self.time_abs_start}' AND time <= '{self.time_abs_end}'"
                    data["range"] = "None"
                else:
                    data["time_abs"] = "None"
                    data["range"] = self.range

                response = requests.post(url,json=data)

                if response.status_code == 200:
                    content = response.content
                    content = StringIO(content.decode("utf-8"))

                    df_temp = pd.read_csv(content, delimiter=self.delimiter) # "read" a csv from the StringIO
                    df_temp[self.cTime] = pd.to_datetime(df_temp[self.cTime], dayfirst=True, format='mixed')

                    # Norm df_temp if flag is set to 'True'
                    if norm:
                        column_name = df_temp.columns[1]
                        temp_max = max(df_temp[column_name])
                        df_temp[column_name] = df_temp[column_name] / temp_max

                    if idx == 0: # First iteration --> write time index once
                        df.insert(0, "Date", df_temp[self.cTime])
                    
                    name_of_payload_col = df_temp.columns[1]
                    df.insert(idx+1, device, df_temp[name_of_payload_col])

                    with open(f"output-iot-server/{filename}", "wb") as f:
                        f.write(response.content)
                    if self.verbose:    
                        tprint(f"Response for '{device}' saved successfully!", "IOT")

                else:
                    tprint(f"Request failed with status code {response.status_code}", "IOT")

                # if idx == 0: # First device --> add values for the prediction horizont for comparison
                #     data["time_abs"] = f"time >= '{self.time_abs_start}' AND time <= '{self.time_abs_end_prediction}'"
                #     response = requests.post(url,json=data)
                #     content = response.content
                #     content = StringIO(content.decode("utf-8"))
                #     df_prediction_horizont = pd.read_csv(content, delimiter=self.delimiter)
                #     df_prediction_horizont[self.cTime] = pd.to_datetime(df_temp[self.cTime], dayfirst=True, format='mixed')
                #     # tprint(df_prediction_horizont)
                #     pass # Download INV values with new time again and append to df, too

                pass
            
            df.set_index("Date", inplace=True)
            # Append df_prediction_horizont here at last column!
            df.drop(df.tail(1).index,inplace=True) # drop last row, since it is always = 0
            df.to_csv("output-iot-server/data_generated.csv", sep=self.delimiter)
            # tprint(df)

            pass
            self.got_df = df
        else:
            self.got_df = None
            tprint(f"IoT Server is not reachable!", "IOT")

        return self.got_df
    
    def prepare_df_for_mpc(self, pred_inv:float, pred_demand:float) -> pd.DataFrame:
        """
        DO NOT USE!
        Prepare the columns of the df for running the MPC algorithm
        --> df should contain at least one BSS and INV2 and INV3
        """
        self.get_df()
        df = self.got_df

        try:
            inv = df.INV1.values + df.INV2.values + df.INV3.values
        except:
            inv = (df.INV2.values + df.INV3.values)/2*3

        demand = df.SHELLY_API_SERVERROOM_POWER.values + df.SHELLY_API_FLOOR_POWER.values
        date = df.index

        data = {"Date":date,
                "inv":inv,
                "demand":demand,
                "pred_inv":pred_inv, # single value
                "pred_demand":pred_demand # single value
                }

        self.df_mpc = pd.DataFrame(data)
        self.df_mpc.set_index("Date", inplace=True)

        return self.df_mpc

    
    def plot(self) -> None:
        if type(self.got_df) != None:
            self.got_df.plot()
            plt.show()

    def setBackendPort(self, port:int) -> None:
        self.backend_port = port

    def setBackendPath(self, path:str) -> None:
        self.backend_path = str(path)

    def setIp(self, ip) -> None:
        self.ip = ip
        self.check_ping()

    def getIp(self) -> str:
        return self.ip

    def setDevices(self, device_list:list) -> None:
        self.devices = device_list
        return None
    
    def activate(self, 
                 inverter=False, 
                 powermeter=False, 
                 bss0=False,
                 bss1=False) -> None:
        """
        Activate DERs of the CoSES lab directly by the name
        """

        if inverter:
            if not "INV2" in self.devices:
                self.devices.append("INV2") 
            if not "INV3" in self.devices:
                self.devices.append("INV3") 

        if powermeter:
            if not "SHELLY_API_FLOOR_POWER" in self.devices:
                self.devices.append("SHELLY_API_FLOOR_POWER") 
            if not "SHELLY_API_SERVERROOM_POWER" in self.devices:
                self.devices.append("SHELLY_API_SERVERROOM_POWER") 

        if bss0:
            if not "E3DC0SOC" in self.devices:
                self.devices.append("E3DC0SOC")  

        if bss1:
            if not "E3DC1SOC" in self.devices:
                self.devices.append("E3DC1SOC") 

        return None

    
    def setTimeAbsStart(self, time_abs_start) -> None:
        self.time_abs_start = time_abs_start
    
    def setTimeAbsEnd(self, time_abs_end) -> None:
        self.time_abs_end = time_abs_end

    def setRange(self, range) -> None:
        """
        In accordance to the backend implementation, is the range dominant
        --> Absolute time can only be used, if range="None"
        """
        self.range = range

    def setRes(self, res:str) -> None:
        # Assign resolution and check, if the unit is valid
        self.res = res
        match = re.match(r"([0-9]+)([a-z]+)", self.res, re.I)
        if match:
            res_number, res_unit = match.groups()
            if len(res_unit) > 1: # Unit is too long (e.g. "min" instead of "m")
                res_unit = res_unit[0]
                self.res = f"{res_number}{res_unit}"

    def setPredictionHorizont(self, horiz) -> None:
        self.prediction_horizont = horiz

    def setPredictionHorizontUnit(self, unit) -> None:
        self.prediction_horizont_unit = unit

    def setDelimiter(self, delimiter) -> None:
        self.delimiter = delimiter

    def timeToStr(self, tstamp)->str:
        tstamp = str(tstamp)
        tstamp = tstamp.replace(" ", "T")
        tstamp = f"{tstamp}Z"
        return tstamp
    
    def getLastValue(self, col_name=""):
        """Get the last value from a col of the df"""
        self.get_df()
        if col_name == "":
            col_name = self.devices[0]
        return self.got_df[col_name].values[-1]


# end IoTGrabber



