from telemetry import tprint # type: ignores

class Battery:

    def __init__(self, capacity, cycle_time, soc_init):
        self.soc = soc_init # [Wh]
        self.capacity = capacity # [Wh]
        self.soc_norm = self.soc / self.capacity # [0..1]
        self.cycle_time = cycle_time # [h]
        print(self)
        pass

    def __str__(self):
        return f"Battery Simulation - Capacity: {self.capacity} Wh, SoC: {self.get_soc_percent()} %"

    def control(self, power_in_w):
        # power_in_w = -power_in_w # Change the sign (POS: Discharge, NEG: Charge)
        charged_energy_for_new_cycle = power_in_w * self.cycle_time # [Wh]
        self.soc += charged_energy_for_new_cycle

        # Limit the energy in the storage
        if self.soc < 0:
            self.soc = 0
        if self.soc > self.capacity:
            self.soc = self.capacity

        self.soc_norm = self.soc / self.capacity
        tprint(f"Capacity: {self.capacity:.2f} Wh, SoC: {self.get_soc_percent():.2f} %, Power: {power_in_w} W", "SIM")
        # print(self)

    def get_soc_norm(self):
        return self.soc_norm
    
    def get_soc_percent(self):
        return self.soc_norm*100
    
    def get_energy_Wh(self):
        return self.soc
    
    def get_energy_kWh(self):
        return self.get_energy_Wh()*1000
    



