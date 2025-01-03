"""
Evaluate a complete cycle of a MPC algorithm and send the values via MQTT

Author:     Benedikt Fuchsgruber
Mail:       benedikt.fuchsgruber@tum.de
Project:    Master's Thesis (MPC for the ZEI DERs)

"""

import paho.mqtt.client as mqtt
import pandas as pd

class Evaluation:
    """
    Evaluate a cycle of the MPC and store the results in the InfluxDB of the IoT-Server
    (Send via MQTT to the backend)

    Parameters:
        - mqtt_broker: Hostname / IP of the MQTT-Broker (here: IoT-Server)
        - mqtt_user: MQTT Username
        - mqtt_password: MQTT Password

    """
    def __init__(self,
                 mqtt_broker:str="10.162.231.9",
                 mqtt_user:str="",
                 mqtt_password:str="",
                 df:pd.DataFrame=pd.DataFrame(),
                 id=0):
        
        
        self.mqtt_broker = mqtt_broker
        self.mqtt_user = mqtt_user
        self.mqtt_password = mqtt_password
        self.df = df
        self.id = id

        self.simulation = 0.0
        
        self.base_topic = f"mpc"
        self.eval_topic = f"{self.base_topic}/eval"

        self.client = mqtt.Client(client_id=f"mpc_eval_{self.id}")
        self.client.username_pw_set(self.mqtt_user, self.mqtt_password)
        self.client.connect(self.mqtt_broker, 1883)
        self.client.publish(f"{self.base_topic}/connect", 1)
        pass

    def cycle (self, df):
        """
        Log values from a pd.DataFrame via MQTT to the defined Broker.
        The column name will be the MQTT topic later

        Parameters:
            - df: pd.DataFrame with different columns, where each col contains a list with ONE value
        """

        # Manipulate the MQTT topic, if simulation is required
        if self.simulation == 1.0:
            self.base_topic = f"mpc_sim"
        else:
            pass


        self.client.connect(self.mqtt_broker, 1883)
        self.data = df.to_dict()
        for key, value in self.data.items():
            topic = f"{self.base_topic}/{key}"
            payload = float(f"{value[0]}")
            self.client.publish(topic, payload)
        pass

    def cycle_without_mpc (self, df):
        """
        Log values from a pd.DataFrame via MQTT to the defined Broker.
        The column name will be the MQTT topic later

        Parameters:
            - df: pd.DataFrame with different columns, where each col contains a list with ONE value
        """
        self.data = df.to_dict()
        for key, value in self.data.items():
            topic = f"{self.base_topic}_without_mpc/{key}"
            payload = float(f"{value[0]}")
            self.client.publish(topic, payload)
        pass

    def __str__(self):
        pass

