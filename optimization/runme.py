import os
import shutil
import urbs
import sys

# Add glpk solver paths to the ENV
solverpath_folder = 'C:\\glpk-4.65\\w64'
solverpath_exe = solverpath_folder
sys.path.append(solverpath_folder)

# Add toolkit path to the ENV
sys.path.append('../toolkit')
sys.path.append('../../toolkit')
from iot_grabber import IotGrabber # type: ignore

iot_server = IotGrabber(ip="100.113.141.113",
                        time_abs_start="2024-09-03T11:00:00Z",
                        time_abs_end="2024-09-05T11:00:00Z",
                        devices=["E3DC0POWER", "INV2", "SHELLY_API_FLOOR_POWER", "SHELLY_API_SERVERROOM_POWER", "E3DC0SOC"],
                        res="15m")

# df = iot_server.get_df(norm=True)
# iot_server.plot()
# print(iot_server)

# input_files = 'single_year_example_mod.xlsx'  # for single year file name, for intertemporal folder name
input_files = '1house-192-kW.xlsx'  # for single year file name, for intertemporal folder name
input_dir = 'Input'
input_path = os.path.join(input_dir, input_files)

result_name = 'Run'
result_dir = urbs.prepare_result_directory(result_name)  # name + time stamp

# copy input file to result directory
try:
    shutil.copytree(input_path, os.path.join(result_dir, input_dir))
except NotADirectoryError:
    shutil.copyfile(input_path, os.path.join(result_dir, input_files))
# copy run file to result directory
shutil.copy(__file__, result_dir)

# objective function
objective = 'cost'  # set either 'cost' or 'CO2' as objective

# Choose Solver (cplex, glpk, gurobi, ...)
solver = 'glpk'

# simulation timesteps
# (offset, length) = (3500, 24)  # time step selection
# Adjust it to 15min steps for 48h --> 192 steps á 15min
(offset, length) = (0, 192) # (0, 24*4*2)
timesteps = range(offset, offset+length)
dt = 1  # length of each time step (unit: hours)


# detailed reporting commodity/sites
report_tuples = [
    (2024, 'coses', 'Electricity')
]

# optional: define names for sites in report_tuples
report_sites_name = {}

# plotting commodities/sites
plot_tuples = []

# optional: define names for sites in plot_tuples
plot_sites_name = {}

# plotting timesteps
plot_periods = {
    'all': timesteps[1:]
}

# add or change plot colors
my_colors = {}
for country, color in my_colors.items():
    urbs.COLORS[country] = color



# for scenario in scenarios:
#     prob = urbs.run_scenario_iot_server(input_path, timesteps, scenario,
#                                         result_dir, dt, objective,
#                                         plot_tuples=plot_tuples,
#                                         plot_sites_name=plot_sites_name,
#                                         plot_periods=plot_periods,
#                                         report_tuples=report_tuples,
#                                         report_sites_name=report_sites_name,
#                                         iot_server=iot_server)

# Start the Urbs optimization for the CoSES
prob, results = urbs.run_scenario_online(   timesteps, 
                                            urbs.scenario_base,
                                            result_dir, 
                                            dt, 
                                            objective,
                                            report_tuples=report_tuples,
                                            report_sites_name=report_sites_name,
                                            df=iot_server.get_df()
                                            )

pass