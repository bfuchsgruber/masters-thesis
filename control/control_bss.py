"""
Python file for the control algorithms related to the E3DC BSS.
The control communicates via MQTT and sends a power target

Author:     Benedikt Fuchsgruber
Mail:       benedikt.fuchsgruber@tum.de
Project:    Master's Thesis (MPC for the ZEI DERs)
"""

from typing import Literal, List
import paho.mqtt.client as mqtt
from telemetry import tprint # type: ignore

import time



class PI():
    """
    Linear PI-Controller with the parameters
        - c_p: Proportional gain [1]
        - c_i: Integration gain [1]
        - t_c: Cycle time of the controller [sec]
    """

    def __init__(self, 
                 c_p=1, 
                 c_i=1, 
                 t_c=1):
        self.c_p = c_p
        self.c_i = c_i
        self.t_c = t_c
        pass



class Control:
    """
    Control algorithms for the E3DC Battery Storage Systems
    Parameters:
        - bss_index: Number of the BSS in charge
    """

    def __init__(self,
                 bss_index:int=Literal[0, 1],
                 power_min_w:float=-6000.0,
                 power_max_w:float= 6000.0,
                 mqtt_broker:str="",
                 mqtt_user:str="",
                 mqtt_password:str="",
                 id=0
                 ):
        
        # Map constructor parameters to instance attributes
        self.bss_index = bss_index
        self.power_min_w = power_min_w
        self.power_max_w = power_max_w
        self.mqtt_broker = mqtt_broker
        self.mqtt_user = mqtt_user
        self.mqtt_password = mqtt_password
        self.id = id

        # Define misc instance attributes
        self.power_target = 0.0
        self.act_soc = 0
        self.simulation = 0.0

        self.power_target_mqtt_topic = f"E3DC/{self.bss_index}/EMS/SET_POWER_VALUE"
        self.soc_mqtt_topic = f"E3DC/{self.bss_index}/EMS/STATEOFCHARGE"
        self.base_topic = f"mpc_bss_control_{self.bss_index}"

        self.client = mqtt.Client(client_id=f"mpcAlgorithmForBss{self.bss_index}_{self.id}")
        self.client.on_message = self.on_message
        self.client.username_pw_set(self.mqtt_user, self.mqtt_password)
        self.client.connect(self.mqtt_broker, 1883)
        self.client.subscribe(self.soc_mqtt_topic)
        self.client.publish(f"{self.base_topic}/connect", 1)

        tprint(self, "BSS")

        pass


    def __str__(self):
        return f"Controller for 'BSS{self.bss_index}' with MQTT-Broker at '{self.mqtt_broker}' initalized"
    

    def on_message(self, msg):
        """
        Callback method for handling incoming MQTT messages by the Paho service
        """
        if msg.topic == self.soc_mqtt_topic:
            value = msg.payload.decode()
            if value:
                self.act_soc = int(value)
                tprint(f"SoC: {self.act_soc}")


    def set_power_target(self, power_target:float, verbose=True) -> None:
        """
        Set a target for the power (signed!) in the range of P_mix / P_max
        """
        self.power_target = int(power_target)

        if self.power_target >= self.power_max_w:
            self.power_target = self.power_max_w
        elif self.power_target <= self.power_min_w:
            self.power_target = self.power_min_w

        if verbose:
            tprint(f" P_target (BSS {self.bss_index}): {self.power_target} W", "BSS")
        pass
    

    def _send_target_to_e3dc(self, verbose=True):
        """
        Send the target to the BSS in charge (whether by the direct control or the PI-Controller)
        """

        # Send power directly to the E3DC
        self.client.publish(self.power_target_mqtt_topic, int(self.power_target))

        # Log command into the controller topic
        self.client.publish(f"{self.base_topic}/{self.power_target_mqtt_topic}", int(self.power_target))

        if verbose:
            tprint("Control target published", "BSS")

        pass


    def direct_control(self, power_target:float, verbose=True) -> None:
        self.set_power_target(power_target, verbose=verbose)

        if self.simulation == 0.0:
            # Send the target only, if no simulation of the BSS is required
            self._send_target_to_e3dc(verbose=verbose)


        pass


    def linear_control(self):
        pi = PI()
        pass

