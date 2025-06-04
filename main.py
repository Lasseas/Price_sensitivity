import numpy as np
import sys
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import os 
import matplotlib.pyplot as plt
import platform
import shutil
import platform
#import psutil
from pyomo.environ import *
#from _run_everything import excel_path, instance, year, num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage
##################################################################
##################################################################
##################################################################
##################################################################

import argparse

#from Generate_data_files import run_everything


parser = argparse.ArgumentParser(description="Run model instance")
#parser.add_argument("--instance", type=int, required=True, help="Instance number (e.g., 1–5)")
parser.add_argument("--year", type=int, required=True, help="Year (e.g., 2025 or 2050)")
parser.add_argument("--carboncost", type=str, required=True, choices=["low", "exp", "high", "zero", "extreme", "psyko"], help="Specify carbon cost level")
#parser.add_argument("--case", type=str, required=True, choices=["wide_small", "wide_medium", "wide_large", "deep_small", "deep_medium", "deep_large", "balanced_small", "balanced_medium", "balanced_large", "max_in", "max_out", "git_push"], help="Specify case type")
#parser.add_argument("--cluster", type=str, required=True, choices=["random", "season", "guided", "diversed"], help="Specify case type")
#parser.add_argument("--industry", type=str, required=True, choices = ["pulp", "alu"], help="Specify industry type")
parser.add_argument("--file", type=str, required=True, help="Path to the Result file")
args = parser.parse_args()

#instance = args.instance

year = args.year
carboncost = args.carboncost
case = "wide_large"
cluster = "season"
filenumber = args.file
industrytype = "pulp"
instance = 1

if industrytype == "pulp":
    excel_path = "NO1_Pulp_Paper_2024_combined historical data_Uten_SatSun.xlsx"
elif industrytype == "alu":
    excel_path = "NO1_Aluminum_2024_combined historical data.xlsx"
else:
    raise ValueError("Invalid industry type. Please choose 'pulp' or 'alu'.")
#excel_path = "NO1_Pulp_Paper_2024_combined historical data_Uten_SatSun.xlsx"
#excel_path = "NO1_Pulp_Paper_2024_combined historical data.xlsx"
#excel_path = "NO1_Aluminum_2024_combined historical data.xlsx"

# Define branch structures for each case type
case_configs = {
    "wide_small": (2, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #128 scenarioer
    "wide_medium": (2, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #512 scenarioer
    "wide_large": (2, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #1058 scenarioer
    "deep_small": (2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0), #128 scenarioer
    "deep_medium": (2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0), #512 scenarioer
    "deep_large": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0), #1024 scenarioer
    "balanced_small": (2, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #128 scenarioer
    "balanced_medium": (2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), #512 scenarioer
    "balanced_large": (2, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0), #1024 scenarioer
    "max_in":  (2, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "max_out":  (2, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "git_push": (2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

(
    num_branches_to_firstStage,
    num_branches_to_secondStage,
    num_branches_to_thirdStage,
    num_branches_to_fourthStage,
    num_branches_to_fifthStage,
    num_branches_to_sixthStage,
    num_branches_to_seventhStage,
    num_branches_to_eighthStage,
    num_branches_to_ninthStage,
    num_branches_to_tenthStage,
    num_branches_to_eleventhStage,
    num_branches_to_twelfthStage,
    num_branches_to_thirteenthStage,
    num_branches_to_fourteenthStage,
    num_branches_to_fifteenthStage
) = case_configs[case]

base_dir = os.path.dirname(os.path.abspath(__file__))

# Build the results folder name using filenumber.
result_folder = os.path.join(base_dir, "Results", f"Results_{filenumber}")
os.makedirs(result_folder, exist_ok=True)

"""
if case != "max_out":

    run_everything(
    excel_path,
    result_folder,
    filenumber,
    instance,
    year,
    cluster,
    num_branches_to_firstStage,
    num_branches_to_secondStage,
    num_branches_to_thirdStage,
    num_branches_to_fourthStage,
    num_branches_to_fifthStage,
    num_branches_to_sixthStage,
    num_branches_to_seventhStage,
    num_branches_to_eighthStage,
    num_branches_to_ninthStage,
    num_branches_to_tenthStage,
    num_branches_to_eleventhStage,
    num_branches_to_twelfthStage,
    num_branches_to_thirteenthStage,
    num_branches_to_fourteenthStage,
    num_branches_to_fifteenthStage
)
   
    
def make_tab_file(filename, data_generator, chunk_size=10_000_000):
        
        #Writes a large dataset to a .tab file in chunks using tab as a delimiter.

        #Parameters:
        #    filename (str): Name of the tab-separated file to save (e.g., 'output.tab').
        #    data_generator (generator): A generator that yields DataFrame chunks.
        #    chunk_size (int): Number of rows to process per chunk.
        
        #first_chunk = True  # Used to write the header only once

        with open(filename, "w", newline='') as f:
            for df_chunk in data_generator:
                df_chunk.to_csv(f, sep = "\t", index=False, header=first_chunk, lineterminator='\n')
                first_chunk = False

        print(f"{filename} saved successfully!")

cost_activity = {
    "Power_Grid": {1: 0, 2: -1.162, 3: 2000, 4: -2000}, # 1 = Import, 2 = Export, 3 = RT_Import, 4 = RT_Export 
    "ElectricBoiler": {1: 0, 2: 0, 3: 0}, #1 = LT, 2 = MT, 3 = Dummy
    "HP_LT": {1: 0, 2: 0}, #1 = LT, 2 = Dummy
    "HP_MT": {1: 0, 2: 0, 3: 0}, #1 = LT, 2 = MT, 3 = Dummy
    "PV" : {1: 0},
    "P2G": {1: 0},
    "G2P": {1: 0},
    "GasBoiler": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, #1 = LT (CH4 mix), 2 = MT (CH4 mix), 3 = LT (CH4), 4 = MT (CH4), 5 = LT (Biogas), 6 = MT (Biogas)
    "GasBoiler_CCS": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, #1 = LT (CH4 mix), 2 = MT (CH4 mix), 3 = LT (CH4), 4 = MT (CH4), 5 = LT (Biogas), 6 = MT (Biogas)
    "CHP": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, #1 = LT (CH4 mix), 2 = MT (CH4 mix), 3 = LT (CH4), 4 = MT (CH4), 5 = LT (Biogas), 6 = MT (Biogas)
    "CHP_CCS": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, #1 = LT (CH4 mix), 2 = MT (CH4 mix), 3 = LT (CH4), 4 = MT (CH4), 5 = LT (Biogas), 6 = MT (Biogas)
    "Biogas_Grid": {1: 64.5, 2: 0}, #1 = Import, 2 = Export
    "CH4_Grid": {1: 39.479, 2: 0}, #1 = Import, 2 = Export
    "CH4_H2_Mixer": {1: 0},
    "DieselReserveGenerator": {1: 148.8},
    "H2_Grid": {1: 150.1502, 2: 0}, #1 = Import, 2 = Export
    "Dummy_Grid": {1: 0} #1 = Export
    }

##################################################################################
############################### READING EXCEL FILE ###############################
##################################################################################

# Function to read all sheets in an Excel file and save each as a .tab file in the current directory
def read_all_sheets(excel):
    # Load the Excel file
    input_excel = pd.ExcelFile(excel)
    
    # Loop over each sheet in the workbook
    for sheet in input_excel.sheet_names:
        # Read the current sheet, skipping the first two rows
        input_sheet = pd.read_excel(excel, sheet_name=sheet, skiprows=2)

        # Drop only fully empty rows (optional)
        data_nonempty = input_sheet.dropna(how='all')

        # Replace spaces in column names with underscores
        data_nonempty.columns = data_nonempty.columns.astype(str).str.replace(' ', '_')

        # Fill missing values with an empty string
        data_nonempty = data_nonempty.fillna('')

        # Convert all columns to strings before replacing whitespace characters in values
        data_nonempty = data_nonempty.applymap(lambda x: str(x) if pd.notnull(x) else "")
        
        # Save as a .tab file using only the sheet name as the file namec
        output_filename = f"{sheet}.tab"
        data_nonempty.to_csv(output_filename, header=True, index=False, sep='\t')
        print(f"Saved file: {output_filename}")

# Call the function with your Excel file
#read_all_sheets('Input_data_With_dummyGrid_and_RT.xlsx')

"""
#####################################################################################
################################ Ble for stor til pushe til git ######################
################################## må genereres i solstorm ##########################
#####################################################################################

#import os
"""
# Always resolve the tab file path relative to script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if case == "max_out":
    if "Pulp" in excel_path:
        tab_file_folder = os.path.join(SCRIPT_DIR, "Out_of_sample")
    elif "Aluminum" in excel_path:
        tab_file_folder = os.path.join(SCRIPT_DIR, "Out_of_sample_alu")
    else:
        raise ValueError("Unknown excel file type. Please check the file name.")
else:
    tab_file_folder = SCRIPT_DIR

# --- Use local folder if max_out, otherwise use script's location ---
import os
if case == "max_out":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    Grandparent_folder = f"Results"
    parent_folder = f"Results_{filenumber}"
    sub_folder = f"Out_of_sample_{filenumber}"

    tab_file_folder = os.path.join(base_dir, Grandparent_folder, parent_folder, sub_folder)
    #tab_file_folder = os.getcwd()  # local working directory (copied out-of-sample folder)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    Grandparent_folder = f"Results"
    parent_folder = f"Results_{filenumber}"
    sub_folder = f"In_sample_data_{filenumber}"

    tab_file_folder = os.path.join(base_dir, Grandparent_folder, parent_folder, sub_folder)
    #tab_file_folder = os.path.dirname(os.path.abspath(__file__))


def generate_cost_activity(num_nodes, num_timesteps, cost_activity, tab_file_folder,  filename="Par_ActivityCost.tab"):
    # Resolve filename relative to the chosen folder
    file_path = os.path.join(tab_file_folder, filename)
   
    def data_generator(chunk_size=10_000_000):
        rows = []
        count = 0
        for node in range(3, num_nodes + 1):
            for tech, mode_costs in cost_activity.items():
                for mode in mode_costs:
                    for t in range(1, num_timesteps + 1):
                        cost = mode_costs[mode]
                        rows.append({
                            "Node": node,
                            "Time": t,
                            "Technology": tech,
                            "Operational_mode": mode,
                            "Cost": cost
                        })
                        count += 1
                        if count % chunk_size == 0:
                            yield pd.DataFrame(rows)
                            rows = []
        if rows:
            yield pd.DataFrame(rows)
   
    make_tab_file(file_path, data_generator())


if case == "max_out":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    Grandparent_folder = f"Results"
    parent_folder = f"Results_{filenumber}"
    sub_folder = f"Out_of_sample_{filenumber}"
    tab_file_folder = os.path.join(base_dir, Grandparent_folder, parent_folder, sub_folder)

    if not os.path.isdir(tab_file_folder):
        raise FileNotFoundError(f"Expected folder does not exist: {tab_file_folder}")
    
    generate_cost_activity(num_nodes= 7812, num_timesteps= 24, tab_file_folder=tab_file_folder, cost_activity=cost_activity) #7812
"""

#####################################################################################
################################## KONSTANTE SETT ###################################
#####################################################################################
#################### HUSK Å ENDRE DISSE I DE ANDRE FILENE OGSÅ ######################
#####################################################################################


####################################################################
######################### MODEL SPECIFICATIONS #####################
####################################################################

model = pyo.AbstractModel()
data = pyo.DataPortal() #Loading the data from a data soruce in a uniform manner (Excel)


"""
SETS 
"""
#Declaring Sets

print("Declaring sets...")

model.Time = pyo.Set(ordered=True) #Set of time periods (hours)
model.Period = pyo.Set(ordered=True) #Set of stages/operational periods
model.LoadShiftingPeriod = pyo.Set(ordered=True) 
#model.LoadShiftingIntervals = pyo.Set(ordered=True)
#model.Time_NO_LoadShift = pyo.Set(dimen = 2, ordered = True) 
#model.TimeLoadShift = pyo.Set(dimen = 3, ordered = True) #Subset of time periods for load shifting in stage s
model.Month = pyo.Set(ordered = True) #Set of months
model.PeriodInMonth = pyo.Set(dimen = 2, ordered = True) #Subset of stages in month m
model.Technology = pyo.Set(ordered = True) #Set of technologies
model.EnergyCarrier = pyo.Set(ordered = True)
model.Mode_of_operation = pyo.Set(ordered = True)
model.TechnologyToEnergyCarrier = pyo.Set(dimen=3, ordered = True)
model.EnergyCarrierToTechnology = pyo.Set(dimen=3, ordered = True)
model.FlexibleLoad = pyo.Set(ordered=True) #Set of flexible loads (batteries)
model.FlexibleLoadForEnergyCarrier = pyo.Set(dimen = 2, ordered = True)
model.Nodes = pyo.Set(ordered=True) #Set of Nodess
model.Nodes_in_stage = pyo.Set(dimen = 2, ordered = True) #Subset of Nodess
model.Nodes_first = pyo.Set(within = model.Nodes) #Subset of Nodess
model.Parent = pyo.Set(ordered=True) #Set of parents
model.Parent_Node = pyo.Set(dimen = 2, ordered = True)


#Reading the Sets, and loading the data
print("Reading sets...")

#data.load(filename="Set_of_TimeSteps_NO_LoadShift.tab", format = "set", set=model.Time_NO_LoadShift)
data.load(filename=os.path.join("Set_of_TimeSteps.tab"), format="set", set=model.Time)
data.load(filename=os.path.join("Set_of_Periods.tab"), format="set", set=model.Period)
data.load(filename=os.path.join("Set_of_LoadShiftingPeriod.tab"), format="set", set=model.LoadShiftingPeriod)
data.load(filename=os.path.join("Set_of_Month.tab"), format="set", set=model.Month)
data.load(filename=os.path.join("Set_of_PeriodsInMonth.tab"), format="set", set=model.PeriodInMonth)
data.load(filename=os.path.join("Set_of_Technology.tab"), format="set", set=model.Technology)
data.load(filename=os.path.join("Set_of_EnergyCarrier.tab"), format="set", set=model.EnergyCarrier)
data.load(filename=os.path.join("Set_Mode_of_Operation.tab"), format="set", set=model.Mode_of_operation)
data.load(filename=os.path.join("Subset_TechToEC.tab"), format="set", set=model.TechnologyToEnergyCarrier)
data.load(filename=os.path.join("Subset_ECToTech.tab"), format="set", set=model.EnergyCarrierToTechnology)
data.load(filename=os.path.join("Set_of_FlexibleLoad.tab"), format="set", set=model.FlexibleLoad)
data.load(filename=os.path.join("Set_of_FlexibleLoadForEC.tab"), format="set", set=model.FlexibleLoadForEnergyCarrier)
data.load(filename=os.path.join("Set_of_Nodes.tab"), format="set", set=model.Nodes)
data.load(filename=os.path.join("Set_of_NodesInStage.tab"), format="set", set=model.Nodes_in_stage)
data.load(filename=os.path.join("Subset_NodesFirst.tab"), format="set", set=model.Nodes_first)
data.load(filename=os.path.join("Set_of_Parents.tab"), format="set", set=model.Parent)
data.load(filename=os.path.join("Set_ParentCoupling.tab"), format="set", set=model.Parent_Node)


"""
PARAMETERS
"""
#Declaring Parameters
print("Declaring parameters...")

#model.Cost_Energy = pyo.Param(model.Nodes, model.Time, model.Technology)  # Cost of using energy source i at time t
model.cost_activity = pyo.Param(model.Nodes, model.Time, model.Technology, model.Mode_of_operation) #Cost of using technology i in mode o at time t
model.Cost_Battery = pyo.Param(model.FlexibleLoad)
#model.Cost_Export = pyo.Param(model.Nodes, model.Time, model.Technology)  # Income from exporting energy to the grid at time t
model.Cost_Expansion_Tec = pyo.Param(model.Technology) #Capacity expansion cost
model.Cost_Expansion_Bat = pyo.Param(model.FlexibleLoad) #Capacity expansion cost
model.Cost_Emission = pyo.Param() #Carbon price
model.Cost_Grid = pyo.Param() #Grid tariff
model.aFRR_Up_Capacity_Price = pyo.Param(model.Nodes, model.Time)  # Capacity Price for aFRR up regulation 
model.aFRR_Dwn_Capacity_Price = pyo.Param(model.Nodes, model.Time)  # Capcaity Price for aFRR down regulation
model.aFRR_Up_Activation_Price = pyo.Param(model.Nodes, model.Time)  # Activation Price for aFRR up regulation 
model.aFRR_Dwn_Activation_Price = pyo.Param(model.Nodes, model.Time)  # Activatioin Price for aFRR down regulation 
model.Spot_Price = pyo.Param(model.Nodes, model.Time)
model.Intraday_Price = pyo.Param(model.Nodes, model.Time)
model.Demand = pyo.Param(model.Nodes, model.Time, model.EnergyCarrier)  # Energy demand 
model.Max_charge_discharge_rate = pyo.Param(model.FlexibleLoad, default = 1) # Maximum symmetric charge and discharge rate
model.Charge_Efficiency = pyo.Param(model.FlexibleLoad)  # Efficiency of charging flexible load b [-]
model.Discharge_Efficiency = pyo.Param(model.FlexibleLoad)  # Efficiency of discharging flexible load b [-]
model.Technology_To_EnergyCarrier_Efficiency = pyo.Param(model.TechnologyToEnergyCarrier) #Efficiency of technology i when supplying fuel e
model.EnergyCarrier_To_Technlogy_Efficiency = pyo.Param(model.EnergyCarrierToTechnology) #Efficiency of technology i when consuming fuel e
model.Max_Storage_Capacity = pyo.Param(model.FlexibleLoad)  # Maximum energy storage capacity of flexible load b [MWh]
model.Self_Discharge = pyo.Param(model.FlexibleLoad)  # Self-discharge rate of flexible load b [%]
model.Initial_SOC = pyo.Param(model.FlexibleLoad)  # Initial state of charge for flexible load b [-]
model.Node_Probability = pyo.Param(model.Nodes)  # Probability of Nodes s [-]
model.Up_Shift_Max = pyo.Param(model.Time)  # Maximum allowable up-shifting in load shifting periods as a percentage of demand [% of demand]
model.Down_Shift_Max = pyo.Param(model.Time)  # Maximum allowable down-shifting in load shifting periods as a percentage of demand [% of demand]
model.Initial_Installed_Capacity = pyo.Param(model.Technology) #Initial installed capacity at site for technology i
model.Ramping_Factor = pyo.Param(model.Technology)
model.Availability_Factor = pyo.Param(model.Nodes, model.Time, model.Technology) #Availability factor for technology delivering to energy carrier 
model.Carbon_Intensity = pyo.Param(model.Technology, model.Mode_of_operation) #Carbon intensity when using technology i in mode o
model.Max_Export = pyo.Param() #Maximum allowable export per year, if no concession is given
model.Activation_Factor_UP_Regulation = pyo.Param(model.Nodes, model.Time) # Activation factor determining the duration of up regulation
model.Activation_Factor_DWN_Regulation = pyo.Param(model.Nodes, model.Time) # Activation factor determining the duration of dwn regulation
model.Activation_Factor_ID_Up = pyo.Param(model.Nodes, model.Time) # Activation factor determining the duration of up regulation
model.Activation_Factor_ID_Dwn = pyo.Param(model.Nodes, model.Time) # Activation factor determining the duration of dwn regulation
model.Available_Excess_Heat = pyo.Param() #Fraction of the total available excess heat at usable temperature level to \\& be used an energy source for the heat pump.
model.Power2Energy_Ratio = pyo.Param(model.FlexibleLoad)
model.Max_CAPEX_tech = pyo.Param(model.Technology)
model.Max_CAPEX_flex = pyo.Param(model.FlexibleLoad)
model.Max_CAPEX = pyo.Param() #Maximum allowable CAPEX
model.Max_Carbon_Emission = pyo.Param() #Maximum allowable carbon emissions per year
model.Last_Period_In_Month = pyo.Param(model.Month) #Last period in month m
model.Cost_LS = pyo.Param(model.EnergyCarrier) #Cost of load shifting for energy carrier e
model.ID_Cap_Buy_volume = pyo.Param(model.Nodes, model.Time) #Volume of ID total bought in the market
model.ID_Cap_Sell_volume = pyo.Param(model.Nodes, model.Time) #Volume of ID total sold in the market
model.Res_Cap_Up_volume = pyo.Param(model.Nodes, model.Time) #Volume of total mFRR up shift in the market
model.Res_Cap_Down_volume = pyo.Param(model.Nodes, model.Time) #Volume of total mFRR down shift in the market

#Reading the Parameters, and loading the data
print("Reading parameters...")

#data.load(filename="Par_EnergyCost.tab", param=model.Cost_Energy, format = "table")
#data.load(filename="Par_ExportCost.tab", param=model.Cost_Export, format = "table")

# Cost Emission - update to load from correct folder
if carboncost == "exp" and year == 2025:
    data.load(filename=os.path.join("Par_CostEmission_exp_2025.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "exp" and year == 2050:
    data.load(filename=os.path.join("Par_CostEmission_exp_2050.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "low" and year == 2025:
    data.load(filename=os.path.join("Par_CostEmission_low_2025.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "low" and year == 2050:
    data.load(filename=os.path.join("Par_CostEmission_low_2050.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "high" and year == 2025:
    data.load(filename=os.path.join("Par_CostEmission_high_2025.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "high" and year == 2050:
    data.load(filename=os.path.join("Par_CostEmission_high_2050.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "zero":
    data.load(filename=os.path.join("Par_CostEmission_zero.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "extreme" and year == 2050:
    data.load(filename=os.path.join("Par_CostEmission_extreme_2050.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "extreme" and year == 2025:
    data.load(filename=os.path.join("Par_CostEmission_extreme_2025.tab"), param=model.Cost_Emission, format = "table")
elif carboncost == "psyko":
    data.load(filename=os.path.join("Par_CostEmission_psyko_extreme_2050.tab"), param=model.Cost_Emission, format = "table")
else:
    raise ValueError("Invalid instance or year. Please check the values.")

#data.load(filename="Par_MaxCableCapacity.tab", param=model.Max_Cable_Capacity, format = "table")
data.load(filename=os.path.join("Par_ActivityCost.tab"), param=model.cost_activity, format="table")
data.load(filename=os.path.join("Par_BatteryCost.tab"), param=model.Cost_Battery, format="table")

if year == 2025:
    data.load(filename=os.path.join("Par_CostExpansion_Tec_2025.tab"), param=model.Cost_Expansion_Tec, format="table")
    data.load(filename=os.path.join("Par_CostExpansion_Bat_2025.tab"), param=model.Cost_Expansion_Bat, format="table")
elif year == 2050:
    data.load(filename=os.path.join("Par_CostExpansion_Tec_2050.tab"), param=model.Cost_Expansion_Tec, format="table")
    data.load(filename=os.path.join("Par_CostExpansion_Bat_2050.tab"), param=model.Cost_Expansion_Bat, format="table")
else:
    raise ValueError("Invalid year. Please check the value.")
# Emission handled later with conditions
data.load(filename=os.path.join("Par_CostGridTariff.tab"), param=model.Cost_Grid, format="table")
data.load(filename=os.path.join("Par_aFRR_UP_CAP_price.tab"), param=model.aFRR_Up_Capacity_Price, format="table")
data.load(filename=os.path.join("Par_aFRR_DWN_CAP_price.tab"), param=model.aFRR_Dwn_Capacity_Price, format="table")
data.load(filename=os.path.join("Par_aFRR_UP_ACT_price.tab"), param=model.aFRR_Up_Activation_Price, format="table")
data.load(filename=os.path.join("Par_aFRR_DWN_ACT_price.tab"), param=model.aFRR_Dwn_Activation_Price, format="table")
data.load(filename=os.path.join("Par_SpotPrice.tab"), param=model.Spot_Price, format="table")
data.load(filename=os.path.join("Par_IntradayPrice.tab"), param=model.Intraday_Price, format="table")
data.load(filename=os.path.join("Par_EnergyDemand.tab"), param=model.Demand, format="table")
data.load(filename=os.path.join("Par_MaxChargeDischargeRate.tab"), param=model.Max_charge_discharge_rate, format="table")
data.load(filename=os.path.join("Par_ChargeEfficiency.tab"), param=model.Charge_Efficiency, format="table")
data.load(filename=os.path.join("Par_DischargeEfficiency.tab"), param=model.Discharge_Efficiency, format="table")
data.load(filename=os.path.join("Par_TechToEC_Efficiency.tab"), param=model.Technology_To_EnergyCarrier_Efficiency, format="table")
data.load(filename=os.path.join("Par_ECToTech_Efficiency.tab"), param=model.EnergyCarrier_To_Technlogy_Efficiency, format="table")
data.load(filename=os.path.join("Par_MaxStorageCapacity.tab"), param=model.Max_Storage_Capacity, format="table")
data.load(filename=os.path.join("Par_SelfDischarge.tab"), param=model.Self_Discharge, format="table")
data.load(filename=os.path.join("Par_InitialSoC.tab"), param=model.Initial_SOC, format="table")
data.load(filename=os.path.join("Par_NodesProbability.tab"), param=model.Node_Probability, format="table")
data.load(filename=os.path.join("Par_MaxUpShift.tab"), param=model.Up_Shift_Max, format="table")
data.load(filename=os.path.join("Par_MaxDwnShift.tab"), param=model.Down_Shift_Max, format="table")
data.load(filename=os.path.join("Par_InitialCapacityInstalled.tab"), param=model.Initial_Installed_Capacity, format="table")
data.load(filename=os.path.join("Par_AvailabilityFactor.tab"), param=model.Availability_Factor, format="table")
data.load(filename=os.path.join("Par_CarbonIntensity.tab"), param=model.Carbon_Intensity, format="table")
data.load(filename=os.path.join("Par_MaxExport.tab"), param=model.Max_Export, format="table")
data.load(filename=os.path.join("Par_ActivationFactor_Up_Reg.tab"), param=model.Activation_Factor_UP_Regulation, format="table")
data.load(filename=os.path.join("Par_ActivationFactor_Dwn_Reg.tab"), param=model.Activation_Factor_DWN_Regulation, format="table")
data.load(filename=os.path.join("Par_ActivationFactor_ID_Up_Reg.tab"), param=model.Activation_Factor_ID_Up, format="table")
data.load(filename=os.path.join("Par_ActivationFactor_ID_Dwn_Reg.tab"), param=model.Activation_Factor_ID_Dwn, format="table")
data.load(filename=os.path.join("Par_AvailableExcessHeat.tab"), param=model.Available_Excess_Heat, format="table")
data.load(filename=os.path.join("Par_Power2Energy_ratio.tab"), param=model.Power2Energy_Ratio, format="table")
data.load(filename=os.path.join("Par_Ramping_factor.tab"), param=model.Ramping_Factor, format="table")
data.load(filename=os.path.join("Par_Max_Capex_tec.tab"), param=model.Max_CAPEX_tech, format="table")
data.load(filename=os.path.join("Par_Max_Capex_bat.tab"), param=model.Max_CAPEX_flex, format="table")
data.load(filename=os.path.join("Par_Max_CAPEX.tab"), param=model.Max_CAPEX, format="table")
data.load(filename=os.path.join("Par_Max_Carbon_Emission.tab"), param=model.Max_Carbon_Emission, format="table")
data.load(filename=os.path.join("Par_LastPeriodInMonth.tab"), param=model.Last_Period_In_Month, format="table")
data.load(filename=os.path.join("Par_Cost_LS.tab"), param=model.Cost_LS, format="table")
data.load(filename=os.path.join("Par_ID_Capacity_Buy_Volume.tab"), param=model.ID_Cap_Buy_volume, format="table")
data.load(filename=os.path.join("Par_ID_Capacity_Sell_Volume.tab"), param=model.ID_Cap_Sell_volume, format="table")
data.load(filename=os.path.join("Par_Res_CapacityUpVolume.tab"), param=model.Res_Cap_Up_volume, format="table")
data.load(filename=os.path.join("Par_Res_CapacityDownVolume.tab"), param=model.Res_Cap_Down_volume, format="table")


"""
VARIABLES
"""
#Declaring Variables
model.x_UP = pyo.Var(model.Nodes, model.Time, domain= pyo.NonNegativeReals)#, bounds = (0,0))
model.x_DWN = pyo.Var(model.Nodes, model.Time, domain= pyo.NonNegativeReals)#, bounds = (0,0))
model.x_DA_buy = pyo.Var(model.Nodes, model.Time, domain= pyo.NonNegativeReals)
model.x_DA_sell = pyo.Var(model.Nodes, model.Time, domain= pyo.NonNegativeReals)
model.x_ID_buy = pyo.Var(model.Nodes, model.Time, domain= pyo.NonNegativeReals)
model.x_ID_sell = pyo.Var(model.Nodes, model.Time, domain= pyo.NonNegativeReals)
model.y_out = pyo.Var(model.Nodes, model.Time, model.TechnologyToEnergyCarrier, domain = pyo.NonNegativeReals)
model.y_in = pyo.Var(model.Nodes, model.Time, model.EnergyCarrierToTechnology, domain = pyo.NonNegativeReals)
model.y_activity = pyo.Var(model.Nodes, model.Time, model.Technology, model.Mode_of_operation, domain = pyo.NonNegativeReals)
model.q_charge = pyo.Var(model.Nodes, model.Time, model.FlexibleLoad, domain= pyo.NonNegativeReals)
model.q_discharge = pyo.Var(model.Nodes, model.Time, model.FlexibleLoad, domain= pyo.NonNegativeReals)
model.q_SoC = pyo.Var(model.Nodes, model.Time, model.FlexibleLoad, domain= pyo.NonNegativeReals)
model.v_new_tech = pyo.Var(model.Technology, domain = pyo.NonNegativeReals)#, bounds = (0,0)) 
model.v_new_bat = pyo.Var(model.FlexibleLoad, domain = pyo.NonNegativeReals)#, bounds = (0,0))
model.y_max = pyo.Var(model.Nodes, model.Month, domain = pyo.NonNegativeReals)
model.d_flex = pyo.Var(model.Nodes, model.Time, model.EnergyCarrier, domain = pyo.NonNegativeReals)
model.Up_Shift = pyo.Var(model.Nodes, model.Time, model.EnergyCarrier, domain = pyo.NonNegativeReals)
model.Dwn_Shift = pyo.Var(model.Nodes, model.Time, model.EnergyCarrier, domain = pyo.NonNegativeReals)
model.aggregated_Up_Shift = pyo.Var(model.Nodes, model.EnergyCarrier, domain = pyo.NonNegativeReals)
model.aggregated_Dwn_Shift = pyo.Var(model.Nodes, model.EnergyCarrier, domain = pyo.NonNegativeReals)
model.Not_Supplied_Energy = pyo.Var(model.Nodes, model.Time, model.EnergyCarrier, domain = pyo.NonNegativeReals)
model.I_loadShedding = pyo.Var()
model.I_inv = pyo.Var()
model.I_GT = pyo.Var()
model.I_cap_bid = pyo.Var(model.Time)
model.I_activation = pyo.Var(model.Nodes, model.Time)
model.I_DA = pyo.Var(model.Nodes, model.Time)
model.I_ID = pyo.Var(model.Nodes, model.Time)
model.I_OPEX = pyo.Var(model.Nodes, model.Time)

#For printout
model.I_cap_bid_printOut = pyo.Var()
model.I_activation_printOut = pyo.Var()
model.I_DA_printOut = pyo.Var()
model.I_ID_printOut = pyo.Var()
model.I_OPEX_printOut = pyo.Var()
model.RealTime_Import = pyo.Var()
model.RealTime_Export = pyo.Var()
model.Dummy_Grid_utilization = pyo.Var()
model.gas_boil_ccs_utilization = pyo.Var()
model.gas_boil_utilization = pyo.Var()
model.el_boil_utilization = pyo.Var()
model.HP_utilization = pyo.Var()
"""
OBJECTIVE
""" 

#OBJECTIVE SHORT FORM
def objective(model):
    obj_expr = model.I_inv + model.I_GT + sum(
        model.I_cap_bid[t] + sum(sum(
            model.Node_Probability[n] * (
                model.I_activation[n, t] + model.I_DA[n, t] + model.I_ID[n, t] + model.I_OPEX[n, t]
            ) for (n, stage) in model.Nodes_in_stage if stage == s
        ) for s in model.Period    
    ) for t in model.Time)

    return obj_expr

model.Objective = pyo.Objective(rule=objective, sense=pyo.minimize)

"""
CONSTRAINTS
"""  

###########################################
############## COST BALANCES ##############
###########################################
def cost_investment(model):
    return model.I_inv == (sum(
        model.Cost_Expansion_Tec[i] * model.v_new_tech[i] for i in model.Technology
    ) + sum(
        model.Cost_Expansion_Bat[b] * model.v_new_bat[b] for b in model.FlexibleLoad
    ))
model.InvestmentCost = pyo.Constraint(rule=cost_investment)

def cost_grid_tariff(model):
    return model.I_GT == sum(sum(model.Node_Probability[n] * model.Cost_Grid * model.y_max[n, m] for (n,s) in model.Nodes_in_stage if s == model.Last_Period_In_Month[m]) for m in model.Month)
model.GridTariffCost = pyo.Constraint(rule=cost_grid_tariff)

def cost_capacity_bid(model, t):
    nodes_in_last_stage = {n for (n, stage) in model.Nodes_in_stage if stage == model.Period.last()}
    
    return model.I_cap_bid[t] == sum(
        model.Node_Probability[n] * (
            - (model.aFRR_Up_Capacity_Price[n, t] * model.x_UP[n, t] +
               model.aFRR_Dwn_Capacity_Price[n, t] * model.x_DWN[n, t])
        ) for n in model.Nodes if n not in nodes_in_last_stage
    )
model.CapacityBidCost = pyo.Constraint(model.Time, rule=cost_capacity_bid)

def cost_activation(model, n, p, t, s):
    if (n, s) in model.Nodes_in_stage:
        return model.I_activation[n, t] == (- model.Activation_Factor_UP_Regulation[n, t] * model.aFRR_Up_Activation_Price[n, t] * model.x_UP[p, t]
                + model.Activation_Factor_DWN_Regulation[n, t] * model.aFRR_Dwn_Activation_Price[n, t] * model.x_DWN[p, t])
    else:
        return pyo.Constraint.Skip
model.ActivationCost = pyo.Constraint(model.Parent_Node, model.Time, model.Period, rule=cost_activation)

def cost_DA(model, n, p, t, s):
    if (n,s) in model.Nodes_in_stage:
        return model.I_DA[n, t] == model.Spot_Price[n, t] * (model.x_DA_buy[p, t] - model.x_DA_sell[p, t])
    else:
        return pyo.Constraint.Skip
model.DACost = pyo.Constraint(model.Parent_Node, model.Time, model.Period, rule=cost_DA) 

def cost_ID(model, n, p, t, s):
    if (n,s) in model.Nodes_in_stage:
        return model.I_ID[n, t] == model.Intraday_Price[n, t] * (
                model.Activation_Factor_ID_Up[n, t] * model.x_ID_buy[p, t] 
                - model.Activation_Factor_ID_Dwn[n, t] * model.x_ID_sell[p, t]
            )
    else:
        return pyo.Constraint.Skip
model.IDCost = pyo.Constraint(model.Parent_Node, model.Time, model.Period, rule=cost_ID)    

def cost_opex(model, n, s, t):
    return model.I_OPEX[n, t] == (sum(
                model.y_activity[n, t, i, o] * (model.cost_activity[n, t, i, o] 
                + model.Carbon_Intensity[i, o] * model.Cost_Emission)
                for (i, e, o) in model.TechnologyToEnergyCarrier 
            ) 
            - sum(model.cost_activity[n, t, i, o] * model.y_activity[n, t, i, o] for (i, e, o) in model.EnergyCarrierToTechnology)
            + sum(model.Cost_Battery[b] * model.q_discharge[n, t, b] for b in model.FlexibleLoad)
            + sum(model.Cost_LS[e]*model.Dwn_Shift[n, t, e] + 10_000 * model.Not_Supplied_Energy[n, t, e] for e in model.EnergyCarrier)
    )
model.OPEXCost = pyo.Constraint(model.Nodes_in_stage, model.Time, rule=cost_opex)



#########################################################################################################
########################### FOR UTSKRIFT AV DE ULIKE OBJEKTIVKOSTNADENE #################################
################################ IKKE LAGT TIL I OBJETIVFUNKSJONEN ######################################
#########################################################################################################

def cost_load_shedding_for_printout(model):
    return model.I_loadShedding == sum(
        model.Node_Probability[n] * 10_000 * model.Not_Supplied_Energy[n, t, e]
        for (n, s) in model.Nodes_in_stage
        for t in model.Time
        for e in model.EnergyCarrier
        if s in model.Period  
    )

model.CostLoadShedding_printout = pyo.Constraint(rule=cost_load_shedding_for_printout)

# I_Inv og I_GT hentes direkte

def cost_capacity_bid_for_printout(model):
    nodes_in_last_stage = {n for (n, stage) in model.Nodes_in_stage if stage == model.Period.last()}
    
    return model.I_cap_bid_printOut == sum(
        model.Node_Probability[n] * (
            - (model.aFRR_Up_Capacity_Price[n, t] * model.x_UP[n, t] +
               model.aFRR_Dwn_Capacity_Price[n, t] * model.x_DWN[n, t])
        )
        for n in model.Nodes if n not in nodes_in_last_stage
        for t in model.Time
    )

model.CapacityBidCost_printout = pyo.Constraint(rule=cost_capacity_bid_for_printout)

def cost_activation_for_printout(model):
    return model.I_activation_printOut == sum(
        model.Node_Probability[n] * (
            - model.Activation_Factor_UP_Regulation[n, t] * model.aFRR_Up_Activation_Price[n, t] * model.x_UP[p, t]
            + model.Activation_Factor_DWN_Regulation[n, t] * model.aFRR_Dwn_Activation_Price[n, t] * model.x_DWN[p, t]
        )
        for t in model.Time
        for s in model.Period
        for (n, stage) in model.Nodes_in_stage if stage == s
        for (n_,p) in model.Parent_Node if n_ == n  
    )
model.ActivationCost_printout = pyo.Constraint(rule=cost_activation_for_printout)

def cost_DA_for_printout(model):
    return model.I_DA_printOut == sum(
        model.Node_Probability[n] * model.Spot_Price[n, t] * (model.x_DA_buy[p, t] - model.x_DA_sell[p, t])
        for t in model.Time
        for s in model.Period
        for (n, stage) in model.Nodes_in_stage if stage == s
        for (n_, p) in model.Parent_Node if n_ == n
    )

model.DACostPrintout = pyo.Constraint(rule=cost_DA_for_printout)

def cost_ID_for_printout(model):
    return model.I_ID_printOut == sum(
        model.Node_Probability[n] * model.Intraday_Price[n, t] * (
            model.Activation_Factor_ID_Up[n, t] * model.x_ID_buy[p, t]
            - model.Activation_Factor_ID_Dwn[n, t] * model.x_ID_sell[p, t]
        )
        for t in model.Time
        for s in model.Period
        for (n, stage) in model.Nodes_in_stage if stage == s
        for (n_, p) in model.Parent_Node if n_ == n
    )

model.IDCostPrintout = pyo.Constraint(rule=cost_ID_for_printout)

def cost_opex_for_printout(model):
    return model.I_OPEX_printOut == sum(
        model.Node_Probability[n] * (
            sum(
                model.y_activity[n, t, i, o] * (
                    model.cost_activity[n, t, i, o] + model.Carbon_Intensity[i, o] * model.Cost_Emission
                )
                for (i, e, o) in model.TechnologyToEnergyCarrier
            )
            - sum(
                model.cost_activity[n, t, i, o] * model.y_activity[n, t, i, o]
                for (i, e, o) in model.EnergyCarrierToTechnology
            )
            + sum(
                model.Cost_Battery[b] * model.q_discharge[n, t, b]
                for b in model.FlexibleLoad
            )
            + sum(
                model.Cost_LS[e] * model.Dwn_Shift[n, t, e]
                for e in model.EnergyCarrier
            )
        )
        for t in model.Time
        for s in model.Period
        for (n, stage) in model.Nodes_in_stage if stage == s
    )

model.OPEXCostPrintout = pyo.Constraint(rule=cost_opex_for_printout)

def real_time_import_cost_rule(model):
    return model.RealTime_Import == sum(
        model.Node_Probability[n] *
        model.y_activity[n, t, "Power_Grid", 3] *
        model.cost_activity[n, t, "Power_Grid", 3]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        if (n, t, "Power_Grid", 3) in model.y_activity and (n, t, "Power_Grid", 3) in model.cost_activity
    )
model.RealTimeImportCostConstraint = pyo.Constraint(rule=real_time_import_cost_rule)

def real_time_export_revenue_rule(model):
    return model.RealTime_Export == - sum(
        model.Node_Probability[n] *
        model.y_activity[n, t, "Power_Grid", 4] *
        model.cost_activity[n, t, "Power_Grid", 4]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        if (n, t, "Power_Grid", 4) in model.y_activity and (n, t, "Power_Grid", 4) in model.cost_activity
    )
model.RealTimeExportRevenueConstraint = pyo.Constraint(rule=real_time_export_revenue_rule)


def dummyfuel_utilization_rule(model):
    return model.Dummy_Grid_utilization == sum(
        model.Node_Probability[n] *
        model.y_in[n, t, "Dummy_Grid", "DummyFuel", o]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        for o in model.Mode_of_operation
        if (n, t, "Dummy_Grid", "DummyFuel", o) in model.y_in
    )
model.DummyFuelUtilizationConstraint = pyo.Constraint(rule=dummyfuel_utilization_rule)

def gasboil_utilization_rule(model):
    return model.gas_boil_utilization == sum(
        model.Node_Probability[n] *
        model.y_out[n, t, "GasBoiler", e, o]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        for e in model.EnergyCarrier if e in ["LT", "MT"]
        for o in model.Mode_of_operation
        if (n, t, "GasBoiler", e, o) in model.y_out
    )
model.GasBoilUtilizationConstraint = pyo.Constraint(rule=gasboil_utilization_rule)

def gasboil_ccs_utilization_rule(model):
    return model.gas_boil_ccs_utilization == sum(
        model.Node_Probability[n] *
        model.y_out[n, t, "GasBoiler_CCS", e, o]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        for e in model.EnergyCarrier if e in ["LT", "MT"]
        for o in model.Mode_of_operation
        if (n, t, "GasBoiler_CCS", e, o) in model.y_out
    )
model.GasBoilCCSUtilizationConstraint = pyo.Constraint(rule=gasboil_ccs_utilization_rule)

def electricboiler_utilization_rule(model):
    return model.el_boil_utilization == sum(
        model.Node_Probability[n] *
        model.y_out[n, t, "ElectricBoiler", e, o]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        for e in model.EnergyCarrier if e in ["LT", "MT"]
        for o in model.Mode_of_operation
        if (n, t, "ElectricBoiler", e, o) in model.y_out
    )
model.ElboilUtilizationConstraint = pyo.Constraint(rule=electricboiler_utilization_rule)


def HP_utilization_rule(model):
    return model.HP_utilization == sum(
        model.Node_Probability[n] *
        model.y_out[n, t, "HP_MT", e, o]
        for (n, stage) in model.Nodes_in_stage
        for t in model.Time
        for e in model.EnergyCarrier if e in ["LT", "MT"]
        for o in model.Mode_of_operation
        if (n, t, "HP_MT", e, o) in model.y_out
    )
model.HPUtilizationConstraint = pyo.Constraint(rule=HP_utilization_rule)


########################################################################################################
########################################################################################################

###########################################
############## ENERGY BALANCE #############
###########################################

def energy_balance(model, n, s, t, e):
    return (
        model.d_flex[n, t, e]
        == sum(sum(model.y_out[n, t, i, e, o] for i in model.Technology if (i,e,o) in model.TechnologyToEnergyCarrier)
        - sum(model.y_in[n, t, i, e, o] for i in model.Technology if(i,e,o) in model.EnergyCarrierToTechnology) for o in model.Mode_of_operation)
        - sum(
            model.Charge_Efficiency[b] * model.q_charge[n, t, b] - model.q_discharge[n, t, b]
            for b in model.FlexibleLoad if (b,e) in model.FlexibleLoadForEnergyCarrier
        )
    )
model.EnergyBalance = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule=energy_balance)

def Defining_flexible_demand(model, n, s, t, e):
    return model.d_flex[n, t, e] == model.Demand[n, t, e] + model.Up_Shift[n, t, e] - model.Dwn_Shift[n, t, e] - model.Not_Supplied_Energy[n, t, e]
model.DefiningFlexibleDemand = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule = Defining_flexible_demand)

#####################################################################################
########################### MARKET BALANCE DA/ID/RT #################################
#####################################################################################

def market_balance_import(model, n, p, t, s, i, e, o):
    if (i, e, o) == ("Power_Grid", "Electricity", 1) and (n,s) in model.Nodes_in_stage:
        return (model.y_out[n, t, i, e, o] == model.x_DA_buy[p, t] + model.Activation_Factor_ID_Up[n,t]*model.x_ID_buy[p, t] + model.Activation_Factor_DWN_Regulation[n, t] * model.x_DWN[p, t])
    else:
        return pyo.Constraint.Skip      
model.MarketBalanceImport = pyo.Constraint(model.Parent_Node, model.Time, model.Period, model.TechnologyToEnergyCarrier, rule = market_balance_import)

def market_balance_export(model, n, p, t, s, i, e, o):
    if (i, e, o) == ("Power_Grid", "Electricity", 2) and (n,s) in model.Nodes_in_stage:
        return (model.y_in[n, t, i, e, o] == model.x_DA_sell[p, t] + model.Activation_Factor_ID_Dwn[n,t]*model.x_ID_sell[p, t] + model.Activation_Factor_UP_Regulation[n, t] * model.x_UP[p, t])
    else:
        return pyo.Constraint.Skip      
model.MarketBalanceExport = pyo.Constraint(model.Parent_Node, model.Time, model.Period, model.EnergyCarrierToTechnology, rule = market_balance_export)

def Max_ID_Buy_Adjustment(model, n, t):
    nodes_in_last_stage = {n for (n, stage) in model.Nodes_in_stage if stage == model.Period.last()}
    if n not in nodes_in_last_stage:
        return (model.x_ID_buy[n, t] <= 0.2*model.ID_Cap_Buy_volume[n, t])
    else:
        return pyo.Constraint.Skip
model.MaxIDBuyAdjustment = pyo.Constraint(model.Nodes, model.Time, rule = Max_ID_Buy_Adjustment)

def Max_ID_Sell_Adjustment(model, n, t):
    nodes_in_last_stage = {n for (n, stage) in model.Nodes_in_stage if stage == model.Period.last()}
    if n not in nodes_in_last_stage:
        return (model.x_ID_sell[n, t] <= 0.2*model.ID_Cap_Sell_volume[n, t])
    else:
        return pyo.Constraint.Skip
model.MaxIDSellAdjustment = pyo.Constraint(model.Nodes, model.Time, rule = Max_ID_Sell_Adjustment)

#####################################################################################
########################### CONVERSION BALANCE ######################################
#####################################################################################

def conversion_balance_out(model, n, s, t, i, e, o):   
    return (model.y_out[n, t, i, e, o] == model.y_activity[n, t, i, o] * model.Technology_To_EnergyCarrier_Efficiency[i, e, o])     
model.ConversionBalanceOut = pyo.Constraint(model.Nodes_in_stage, model.Time, model.TechnologyToEnergyCarrier, rule = conversion_balance_out)

def conversion_balance_in(model, n, s, t, i, e, o):
    return (model.y_in[n, t, i, e, o] == model.y_activity[n, t, i, o] * model.EnergyCarrier_To_Technlogy_Efficiency[i, e, o])           
model.ConversionBalanceIn = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrierToTechnology, rule = conversion_balance_in)

#####################################################################################
########################### TECHNOLOGY RAMPING CONSTRAINTS ##########################
#####################################################################################
"""
def Ramping_Technology(model, n, p, t, s, i, e, o):
    if (n,s) in model.Nodes_in_stage:
        if t == model.Time.first() and s == model.Period.first(): #Første tidssteg i første stage  
            return (model.y_out[n, t, i, e, o] <= model.Ramping_Factor[i] * (model.Initial_Installed_Capacity[i] + model.v_new_tech[i]))
        
        elif t == model.Time.first() and s > model.Period.first():
            return (model.y_out[n, t, i, e, o] - model.y_out[p, model.Time.last(), i, e, o] <= model.Ramping_Factor[i] * (model.Initial_Installed_Capacity[i] + model.v_new_tech[i]))

        else:
            return (model.y_out[n, t, i, e, o] - model.y_out[n, t-1, i, e, o] <= model.Ramping_Factor[i] * (model.Initial_Installed_Capacity[i] + model.v_new_tech[i]))
    else:
        return pyo.Constraint.Skip
model.RampingTechnology = pyo.Constraint(model.Parent_Node, model.Time, model.Period, model.TechnologyToEnergyCarrier, rule = Ramping_Technology)

"""

#####################################################################################
############## HEAT PUMP LIMITATION - MÅ ENDRES I HENHOLD TIL INPUTDATA #############
#####################################################################################
"""
def heat_pump_input_limitation_LT(model, n, s, t):
    return (
        model.y_out[n, t, 'HP_LT', 'LT', 1] - model.y_in[n, t, 'HP_LT', 'Electricity', 1]
        <= model.Available_Excess_Heat * (model.d_flex[n, t, 'LT'])# + model.Demand[s, t, 'HT'])
    )
model.HeatPumpInputLimitationLT = pyo.Constraint(model.Nodes_in_stage, model.Time, rule=heat_pump_input_limitation_LT)

def heat_pump_input_limitation_MT(model, n, s, t):
    return (
        model.y_out[n, t, 'HP_MT', 'MT', 1] - model.y_in[n, t, 'HP_MT', 'Electricity', 1]
        <= model.Available_Excess_Heat * (model.d_flex[n, t, 'MT'])# + model.Demand[s, t, 'HT'])
    )
model.HeatPumpInputLimitationMT = pyo.Constraint(model.Nodes_in_stage, model.Time, rule=heat_pump_input_limitation_MT)
"""

def heat_pump_input_limitation(model, n, s, t):
    return (
        model.y_out[n, t, 'HP_MT', 'MT', 1] - model.y_in[n, t, 'HP_MT', 'Electricity', 1] 
        + model.y_out[n, t, 'HP_MT', 'LT', 2] - model.y_in[n, t, 'HP_MT', 'Electricity', 2] 
        + model.y_out[n, t, 'HP_LT', 'LT', 1] - model.y_in[n, t, 'HP_LT', 'Electricity', 1]
        <= model.Available_Excess_Heat * (model.d_flex[n, t, 'LT'] + model.d_flex[n, t, 'MT'])
    )
model.HeatPumpInputLimitation = pyo.Constraint(model.Nodes_in_stage, model.Time, rule=heat_pump_input_limitation)


######################################################
############## LOAD SHIFTING CONSTRAINTS #############
######################################################

def aggregated_up_shift(model, n, p, e):
    return model.aggregated_Up_Shift[n, e] == model.aggregated_Up_Shift[p, e] + sum(model.Up_Shift[n, t, e] for t in model.Time)
model.AggregatedUpShift = pyo.Constraint(model.Parent_Node, model.EnergyCarrier, rule=aggregated_up_shift)

def aggregated_dwn_shift(model, n, p, e):
    return model.aggregated_Dwn_Shift[n, e] == model.aggregated_Dwn_Shift[p, e] + sum(model.Dwn_Shift[n, t, e] for t in model.Time)
model.AggregatedDwnShift = pyo.Constraint(model.Parent_Node, model.EnergyCarrier, rule=aggregated_dwn_shift)

def balancing_aggregated_shifted_load(model, n, s, e):
    if s in model.LoadShiftingPeriod:
        return model.aggregated_Up_Shift[n, e] == model.aggregated_Dwn_Shift[n, e]
    else:
        return pyo.Constraint.Skip
model.BalancingAggregatedShiftedLoad = pyo.Constraint(model.Nodes_in_stage, model.EnergyCarrier, rule=balancing_aggregated_shifted_load)

def initialize_aggregated_up_shift(model, n, e):
    return model.aggregated_Up_Shift[n, e] == 0
model.InitializeAggregatedUpShift = pyo.Constraint(model.Nodes_first, model.EnergyCarrier, rule=initialize_aggregated_up_shift)

def initialize_aggregated_dwn_shift(model, n, e):
    return model.aggregated_Dwn_Shift[n, e] == 0
model.InitializeAggregatedDwnShift = pyo.Constraint(model.Nodes_first, model.EnergyCarrier, rule=initialize_aggregated_dwn_shift)

"""
def No_Up_Shift_outside_window(model, n, s, t, e):
    if (t,s) in model.Time_NO_LoadShift:
        return model.Up_Shift[n, t, e] == 0
    else:
        return pyo.Constraint.Skip
model.NoUpShiftOutsideWindow = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule=No_Up_Shift_outside_window)

def No_Dwn_Shift_outside_window(model, n, s, t, e):
    if (t,s) in model.Time_NO_LoadShift:
        return model.Dwn_Shift[n, t, e] == 0
    else:
        return pyo.Constraint.Skip
model.NoDwnShiftOutsideWindow = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule=No_Dwn_Shift_outside_window)
"""

###########################################################
############## MAX ALLOWABLE UP/DOWN SHIFT ################
###########################################################

def max_up_shift(model, n, s, t, e):
    return model.Up_Shift[n, t, e] <= model.Up_Shift_Max[t] * model.Demand[n, t, e]    
model.MaxUpShift = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule=max_up_shift)

def max_dwn_shift(model, n, s, t, e):
    return model.Dwn_Shift[n, t, e] <= model.Down_Shift_Max[t] * model.Demand[n, t, e]
model.MaxDwnShift = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule=max_dwn_shift)

"""
def Max_total_up_dwn_load_shift(model, n, s, t, e):
    return model.Up_Shift[n,t,e] + model.Dwn_Shift[n,t,e] <= model.Up_Shift_Max * model.Demand[n, t, e] 
model.MaxTotalUpDwnLoadShift = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrier, rule=Max_total_up_dwn_load_shift)
"""

########################################################################
############## RESERVE MARKET PARTICIPATION LIMITS #####################
########################################################################
"""
def reserve_down_limit(model, n, p, t, s, e):
    if e == "Electricity" and (n,s) in model.Nodes_in_stage:  # Ensure e = EL
        return model.x_DWN[p, t] <= (
            model.Up_Shift_Max[t] * model.Demand[n, t, e]
            + sum(
                model.Max_charge_discharge_rate[b] + model.Power2Energy_Ratio[b] * model.v_new_bat[b]
                for b in model.FlexibleLoad if (b, e) in model.FlexibleLoadForEnergyCarrier
            )
        )
    else:
        return pyo.Constraint.Skip
model.ReserveDownLimit = pyo.Constraint(model.Parent_Node, model.Time, model.Period, model.EnergyCarrier, rule=reserve_down_limit)

def reserve_up_limit(model, n, p, t, s, e):
    if e == "Electricity" and (n,s) in model.Nodes_in_stage:  # Ensure e = EL
        return model.x_UP[p, t] <= (
            model.Down_Shift_Max[t] * model.Demand[n, t, e]
            + sum(
                model.Max_charge_discharge_rate[b] + model.Power2Energy_Ratio[b] * model.v_new_bat[b]
                for b in model.FlexibleLoad if (b, e) in model.FlexibleLoadForEnergyCarrier
            )
        )
    else:
        return pyo.Constraint.Skip
model.ReserveUpLimit = pyo.Constraint(model.Parent_Node, model.Time, model.Period, model.EnergyCarrier, rule=reserve_up_limit)
"""
########################################################################
############## UPPER-UPPER BOUND CAPACITY MARKET BIDS ##################
########################################################################

def max_capacity_up_bid(model, n, t):
    return model.x_UP[n,t] <= min(50, 0.2*model.Res_Cap_Up_volume[n,t])
model.MaxCapacityUpBid = pyo.Constraint(model.Nodes, model.Time, rule=max_capacity_up_bid)

def max_capacity_down_bid(model, n, t):
    return model.x_DWN[n,t] <= min(50, 0.2*model.Res_Cap_Down_volume[n,t])
model.MaxCapacityDownBid = pyo.Constraint(model.Nodes, model.Time, rule=max_capacity_down_bid)
"""
def maximum_market_down_reserve_limit(model, n, t):
    return model.x_DWN[n,t] <= 0.2*model.Res_Cap_Down_volume[n,t] #Limiting 
model.MaxMarketDownReserveLimit = pyo.Constraint(model.Nodes, model.Time, rule=maximum_market_down_reserve_limit)

def maximum_market_up_reserve_limit(model, n, t):
    return model.x_UP[n,t] <= 0.2*model.Res_Cap_Up_volume[n,t]
model.MaxMarketUpReserveLimit = pyo.Constraint(model.Nodes, model.Time, rule=maximum_market_up_reserve_limit)
"""
########################################################################
############## FLEXIBLE ASSET CONSTRAINTS/STORAGE DYNAMICS #############
########################################################################
def flexible_asset_charge_discharge_limit(model, n, s, t, b, e):
    return (
        model.q_charge[n, t, b] 
        + model.q_discharge[n, t, b] / model.Discharge_Efficiency[b] 
        <= model.Max_charge_discharge_rate[b] + model.Power2Energy_Ratio[b] * model.v_new_bat[b]
    )
model.FlexibleAssetChargeDischargeLimit = pyo.Constraint(model.Nodes_in_stage, model.Time, model.FlexibleLoadForEnergyCarrier, rule=flexible_asset_charge_discharge_limit)

def state_of_charge(model, n, p, t, s, b, e):
    if (n,s) in model.Nodes_in_stage:
        if t == model.Time.first() and s == model.Period.first() :  # Initialisation of flexible assets
            return (
                model.q_SoC[n, t, b]
                == model.Initial_SOC[b] * (model.Max_Storage_Capacity[b] + model.v_new_bat[b]) * (1 - model.Self_Discharge[b])
                + model.q_charge[n, t, b]
                - model.q_discharge[n, t, b] / model.Discharge_Efficiency[b]
            )
        elif t == model.Time.first() and s > model.Period.first():  #Overgangen mellom stages
            return (
                model.q_SoC[n, t, b]
                == model.q_SoC[p, model.Time.last(), b] * (1 - model.Self_Discharge[b])
                + model.q_charge[n, t, b]
                - model.q_discharge[n, t, b] / model.Discharge_Efficiency[b]
            )
        else:        
            return (
                model.q_SoC[n, t, b]
                == model.q_SoC[n, t-1, b] * (1 - model.Self_Discharge[b])
                + model.q_charge[n, t, b]
                - model.q_discharge[n, t, b] / model.Discharge_Efficiency[b]
            )
    else:
        return pyo.Constraint.Skip
model.StateOfCharge = pyo.Constraint(model.Parent_Node, model.Time, model.Period, model.FlexibleLoadForEnergyCarrier, rule=state_of_charge)

def end_of_horizon_SoC(model, n, s, t, b, e):
    if t == model.Time.last() and s == model.Period.last():
        return model.q_SoC[n, t, b] == model.Initial_SOC[b] * (model.Max_Storage_Capacity[b] + model.v_new_bat[b])
    else:
        return pyo.Constraint.Skip
model.EndOfHorizonSoC = pyo.Constraint(model.Nodes_in_stage, model.Time, model.FlexibleLoadForEnergyCarrier, rule = end_of_horizon_SoC)

def flexible_asset_energy_limit(model, n, s, t, b, e):
    return model.q_SoC[n, t, b] <= model.Max_Storage_Capacity[b] + model.v_new_bat[b]
model.FlexibleAssetEnergyLimits = pyo.Constraint(model.Nodes_in_stage, model.Time, model.FlexibleLoadForEnergyCarrier, rule=flexible_asset_energy_limit)

####################################################
############## AVAILABILITY CONSTRAINT #############
####################################################

def supply_limitation(model, n, s, t, i):
    return (sum(model.y_out[n, t, i, e, o] for e,o in model.EnergyCarrier * model.Mode_of_operation if (i,e,o) in model.TechnologyToEnergyCarrier)  
                <= model.Availability_Factor[n, t, i] * (model.Initial_Installed_Capacity[i] + model.v_new_tech[i]))
model.SupplyLimitation = pyo.Constraint(model.Nodes_in_stage, model.Time, model.Technology, rule=supply_limitation)

##############################################################
############## EXPORT LIMITATION AND GRID TARIFF #############
##############################################################
"""
def export_limitation(model, n, s, t, i, e, o):
    if (i, e, o) == ('Power_Grid', 'Electricity', 2):
        return model.y_in[n, t, i, e, o] <= model.Max_Export
    else:
        return pyo.Constraint.Skip
model.ExportLimitation = pyo.Constraint(model.Nodes_in_stage, model.Time, model.EnergyCarrierToTechnology, rule=export_limitation)
"""
def peak_load(model, n, s, t, m, i, e, o):
    if i == 'Power_Grid' and e == 'Electricity' and (m,s) in model.PeriodInMonth:
        return sum(model.y_out[n, t, i, e, o] for o in model.Mode_of_operation if (i,e,o) in model.TechnologyToEnergyCarrier) <= model.y_max[n, m]
    else:
        return pyo.Constraint.Skip
model.PeakLoad = pyo.Constraint(model.Nodes_in_stage, model.Time, model.Month, model.TechnologyToEnergyCarrier, rule=peak_load)

def Node_greater_than_parent(model, n, p, s, m):
    """
    if (n,s) in model.Nodes_in_stage and (m,s) in model.PeriodInMonth:
        return model.y_max[p, m] <= model.y_max[n, m]
    else:
        return pyo.Constraint.Skip
    """
    # n i stage s og måned m
    if (n, s) in model.Nodes_in_stage and (m, s) in model.PeriodInMonth:
        # Finn alle s_p der p er i den samme måneden
        for s_p in model.Period:
            if (p, s_p) in model.Nodes_in_stage and (m, s_p) in model.PeriodInMonth:
                return model.y_max[p, m] <= model.y_max[n, m]
    return pyo.Constraint.Skip
model.NodeGreaterThanParent = pyo.Constraint(model.Parent_Node, model.Period, model.Month, rule = Node_greater_than_parent)

##############################################################
##################### INVESTMENT LIMITATIONS #################
##############################################################
"""
def CAPEX_technology_limitations(model, i):
    return (model.Cost_Expansion_Tec[i] * model.v_new_tech[i] <= model.Max_CAPEX_tech[i])
model.CAPEXTechnologyLim = pyo.Constraint(model.Technology, rule=CAPEX_technology_limitations)

def CAPEX_flexibleLoad_limitations(model, b):
    return (model.Cost_Expansion_Bat[b] * model.v_new_bat[b] <= model.Max_CAPEX_flex[b])
model.CAPEXFlexibleLoadLim = pyo.Constraint(model.FlexibleLoad, rule=CAPEX_flexibleLoad_limitations)
"""
def CAPEX_limitations(model):
    return model.I_inv <= model.Max_CAPEX
model.CAPEXLim = pyo.Constraint(rule=CAPEX_limitations)

def No_PV_investment(model):
        return model.v_new_tech['PV'] == 0
model.NoPVInvestment = pyo.Constraint(rule=No_PV_investment)

##############################################################
##################### CARBON EMISSION LIMIT ##################
##############################################################
"""
def Carbon_Emission_Limit(model, n): #Kan løses med aggregert variabel og parent-nodes
    total_emission = sum(
        model.y_activity[n, t, i, o] * model.Carbon_Intensity[i, o]
        for t in model.Time
        for (i,e,o) in model.TechnologyToEnergyCarrier
    )
    return total_emission <= model.Max_Carbon_Emission
model.CarbonEmissionLimit = pyo.Constraint(model.Nodes_in_stage, rule=Carbon_Emission_Limit)
"""
"""
def Carbon_Emission_Limit(model, n, s): 
    return sum(sum(sum(
        model.y_activity[n, t, i, o] * model.Carbon_Intensity[i, o]
        for o in model.Mode_of_operation if (i,o) in model.Carbon_Intensity) for i in model.Technology) for t in model.Time) <= model.Max_Carbon_Emission
model.CarbonEmissionLimit = pyo.Constraint(model.Nodes_in_stage, rule=Carbon_Emission_Limit)

"""

print("Objective and constraints read...")

"""
MATCHING DATA FROM CASE WITH MATHEMATICAL MODEL AND PRINTING DATA
"""
print("Building instance...")

our_model = model.create_instance(data)  

if case == "max_out":
    print("🔒 Fixing v_new_tech and v_new_bat to 0 for out-of-sample run...")

    for tech in our_model.Technology:
        our_model.v_new_tech[tech].fix(0)

    for bat in our_model.FlexibleLoad:
        our_model.v_new_bat[bat].fix(0)


if case != "max_out":
    print("🔒 Fixing Not_Supplied_Energy to 0 for in-sample run...")
    for n in our_model.Nodes:
        for t in our_model.Time:
            for e in our_model.EnergyCarrier:
                our_model.Not_Supplied_Energy[n, t, e].fix(0)
    



our_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT) #Import dual values into solver results
#import pdb; pdb.set_trace()


"""
SOLVING PROBLEM
"""

import pyomo.common.tempfiles as tempfiles
import os

# Create a local temp folder for Pyomo to avoid shared /tmp conflicts
custom_tmp_dir = os.path.join(os.getcwd(), "pyomo_temp")
os.makedirs(custom_tmp_dir, exist_ok=True)
tempfiles.TempfileManager.tempdir = custom_tmp_dir


print("Solving...")

# === Create Results folder ===


opt = SolverFactory("gurobi", Verbose=True)
opt.options["Crossover"] = 0  # Set crossover value
opt.options["Method"] = 2  # Use the barrier method


import os
import datetime

# === Generate or load shared run label ===
# Only create a new run_label if it's not a max_out case

# Instead of:
# base_dir = os.getcwd()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_label = f"case_{case}_cluster_{cluster}_year_{year}_{timestamp}"


# Use the script’s location as base_dir.
#base_dir = os.path.dirname(os.path.abspath(__file__))

# Build the results folder name using filenumber.
#result_folder = os.path.join(base_dir, "Results", f"Results_{filenumber}")
#os.makedirs(result_folder, exist_ok=True)

# Create a subfolder for input data.
#input_data_folder = os.path.join(result_folder, "input_data")
#os.makedirs(input_data_folder, exist_ok=True)


# Create results folder using a fixed base directory
#result_folder = os.path.join(base_dir, "Results", f"Results_{filenumber}")
#input_data_folder = os.path.join(result_folder, "input_data")
#os.makedirs(input_data_folder, exist_ok=True)
#result_folder = os.path.join("Results", f"Results_{filenumber}")
#in_sample_folder = os.path.join(top_level_results_folder, "In_sample_results")
#out_of_sample_folder = os.path.join(top_level_results_folder, "Out_of_sample_results
#input_data_folder = os.path.join(result_folder, "input_data")



# Instead of:
# result_folder = os.path.join("Results", f"Results_{filenumber}")
# Use:
#result_folder = os.path.join(base_dir, "Results", f"Results_{filenumber}")
#input_data_folder = os.path.join(result_folder, "input_data")
#os.makedirs(input_data_folder, exist_ok=True)


# Subfolders inside it
#in_sample_folder = os.path.join(top_level_results_folder, "In_sample_results")
#out_of_sample_folder = os.path.join(top_level_results_folder, "Out_of_sample_results")
#input_data_folder = os.path.join(top_level_results_folder, "input_data")

# Create only if they don’t exist
#os.makedirs(in_sample_folder, exist_ok=True)
#os.makedirs(out_of_sample_folder, exist_ok=True)



# === Clean up old Gurobi logs ===
#for f in os.listdir(in_sample_folder):
   # if f.startswith("gurobi_log_") and f.endswith(".txt"):
    #    os.remove(os.path.join(in_sample_folder, f))

# === Set Gurobi log file in result_folder ===
# Optional cleanup
for f in os.listdir(result_folder):
    if f.startswith("gurobi_log_") and f.endswith(".txt"):
        os.remove(os.path.join(result_folder, f))

# Set temporary Gurobi log
logfile_temp = os.path.join(result_folder, 'gurobi_log_temp.txt')
opt.options['LogFile'] = logfile_temp


print("✅ Created folders:")
print("  Results:", os.path.exists(result_folder))
#print("  Input data:", os.path.exists(input_data_folder))

# === Copy input files ===
#input_extensions = (".tab", ".xlsx", ".csv", ".dat")
#for fname in os.listdir("."):
#    if os.path.isfile(fname) and not fname.endswith(".py") and fname.endswith(input_extensions):
#        shutil.copy2(fname, os.path.join(input_data_folder, fname))

# === Solve model ===
start_time = time.time()
results = opt.solve(our_model, tee=True)
end_time = time.time()
running_time = end_time - start_time

# === Rename Gurobi log file ===
runtime_str = f"{running_time:.2f}s".replace('.', '_')
#result_target_folder = out_of_sample_folder if case == "max_out" else in_sample_folder
final_logfile = os.path.join(result_folder, f"gurobi_log_{run_label}_{runtime_str}.txt")
os.rename(logfile_temp, final_logfile)


# Optional: Append Python timing info at the bottom
with open(final_logfile, 'a') as f:
    f.write("\n==================== PYTHON TIMING INFO ====================\n")
    f.write(f"Total solving time measured in Python: {running_time:.2f} seconds\n")


# Extract Gurobi solver information
solver_stats = results.solver

# Get the number of simplex iterations
simplex_iterations = solver_stats.statistics.number_of_iterations if hasattr(solver_stats.statistics, 'number_of_iterations') else "Unavailable"

"""
DISPLAY RESULTS??
"""
#our_model.display('results.csv')
#our_model.dual.display()
print("-" * 70)
print("Objective and running time:")
print(f"Objective value: {round(pyo.value(our_model.Objective),2)}")
print(f"The instance was solved in {round(running_time, 4)} seconds🙂")
print("-" * 70)
print("Hardware details:")
print(f"Processor: {platform.processor()}")
print(f"Machine: {platform.machine()}")
print(f"System: {platform.system()} {platform.release()}")
#print(f"CPU Cores: {psutil.cpu_count(logical=True)} (Logical), {psutil.cpu_count(logical=False)} (Physical)")
#print(f"Total Memory: {psutil.virtual_memory().total / 1e9:.2f} GB")
print("-" * 70)
#import pdb; pdb.set_trace()



# === Write runtime info to .txt ===
runtime_log = f"""Solver Runtime Log
--------------------
Total Solving Time (end time - start time): {running_time:.2f} seconds

Simplex Iterations: {simplex_iterations}
"""

runtime_txt_filename = f"runtime_log_{run_label}.txt"
with open(os.path.join(result_folder, runtime_txt_filename), "w") as f:
    f.write(runtime_log)


"""
EXTRACT VALUE OF VARIABLES AND WRITE THEM INTO EXCEL FILE
"""

print("Writing results to .xlsx...")

def save_results_to_excel(model_instance, run_label, target_folder, max_rows_per_sheet=1_000_000):
    import pandas as pd
    from pyomo.environ import value
    import os

    filename = f"Variable_Results_{run_label}.xlsx"
    full_path = os.path.join(target_folder, filename)

    try:
        import xlsxwriter
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "xlsxwriter"])
        import site
        site.ENABLE_USER_SITE = True
        site.addsitedir(site.getusersitepackages())
        import xlsxwriter

    with pd.ExcelWriter(full_path, engine="xlsxwriter") as writer:
        for var in model_instance.component_objects(pyo.Var, active=True):
            #if var.name not in ["v_new_tech", "v_new_bat", "Not_Supplied_Energy", "I_inv", "I_GT", "I_cap_bid_printOut", "I_loadShedding", "I_activation_printOut", "I_DA_printOut", "I_ID_printOut", "I_OPEX_printOut"]:
            #    continue
            var_name = var.name
            var_data = []

            for index in var:
                try:
                    var_value = value(var[index])
                except:
                    continue
                if abs(var_value) > 1e-3:
                    var_data.append((index, var_value))

            if var_data:
                df = pd.DataFrame(var_data, columns=["Index", var_name])
                max_index_len = max(len(idx) if isinstance(idx, tuple) else 1 for idx, _ in var_data)
                unpacked = pd.DataFrame(
                    [list(idx) + [None] * (max_index_len - len(idx)) if isinstance(idx, tuple) else [idx] for idx, _ in var_data],
                    columns=[f"Index_{i+1}" for i in range(max_index_len)]
                )
                df = pd.concat([unpacked, df[var_name]], axis=1)

                for i in range(0, len(df), max_rows_per_sheet):
                    df_chunk = df.iloc[i:i + max_rows_per_sheet]
                    sheet_title = f"{var_name[:25]}_{i // max_rows_per_sheet + 1}"
                    df_chunk.to_excel(writer, sheet_name=sheet_title, index=False)

    print(f"Variable results saved to {full_path}")
    return full_path


# === Write Excel results ===
excel_filename = save_results_to_excel(our_model, run_label, result_folder)



# === Write case and objective summary ===

# Get the objective value
objective_value = pyo.value(our_model.Objective)
num_Nodes = len(our_model.Nodes) if hasattr(our_model, "Nodes") else "Unknown"
num_days = len(our_model.Period)
num_timesteps = len(our_model.Time)
objective_scaled_to_year = (objective_value / num_days) * 365
investment_cost = pyo.value(our_model.I_inv)
investment_cost_scaled_to_year = (investment_cost/num_days) * 365
loadShedding_cost = pyo.value(our_model.I_loadShedding)
loadShedding_cost_scaled_to_year = (loadShedding_cost/num_days) * 365

Revenue_capacity_market = pyo.value(our_model.I_cap_bid_printOut)
Revenue_activation_market = pyo.value(our_model.I_activation_printOut)
Cost_DA_market = pyo.value(our_model.I_DA_printOut)
Cost_ID_market = pyo.value(our_model.I_ID_printOut)
OPEX_cost = pyo.value(our_model.I_OPEX_printOut)
GridTariff_cost = pyo.value(our_model.I_GT)


Imbalance_cost_import = pyo.value(our_model.RealTime_Import)
Imbalance_cost_export = pyo.value(our_model.RealTime_Export)
DummyFuel_utilization = pyo.value(our_model.Dummy_Grid_utilization)
gasboil_utilization = pyo.value(our_model.gas_boil_utilization)
gasboil_ccs_utilization = pyo.value(our_model.gas_boil_ccs_utilization)
elboil_utilization = pyo.value(our_model.el_boil_utilization)
HP_utilization = pyo.value(our_model.HP_utilization)





if case != "max_out":
    investment_cost_for_out_of_sample = (pyo.value(our_model.I_inv) / num_days) * 5
    investment_cost_scaled_to_year_for_out_of_sample = (investment_cost_for_out_of_sample / 5) * 365

    # Save to file
    with open(os.path.join(result_folder, "in_sample_investment_cost.txt"), "w") as f:
        f.write(f"{investment_cost_for_out_of_sample},{investment_cost_scaled_to_year_for_out_of_sample}")

if case == "max_out":
    # Load the in-sample investment cost from file
    try:
        with open(os.path.join(result_folder, "in_sample_investment_cost.txt"), "r") as f:
            line = f.readline().strip()
            investment_cost_for_out_of_sample, investment_cost_scaled_to_year_for_out_of_sample = map(float, line.split(","))
    except FileNotFoundError:
        print("⚠️ Warning: Could not find in_sample_investment_cost.txt. Defaulting investment cost to 0.")
        investment_cost_for_out_of_sample = 0.0
        investment_cost_scaled_to_year_for_out_of_sample = 0.0

        
# List of your branch counts
branches = [
    num_branches_to_firstStage,
    num_branches_to_secondStage,
    num_branches_to_thirdStage,
    num_branches_to_fourthStage,
    num_branches_to_fifthStage,
    num_branches_to_sixthStage,
    num_branches_to_seventhStage,
    num_branches_to_eighthStage,
    num_branches_to_ninthStage,
    num_branches_to_tenthStage,
    num_branches_to_eleventhStage,
    num_branches_to_twelfthStage,
    num_branches_to_thirteenthStage,
    num_branches_to_fourteenthStage,
    num_branches_to_fifteenthStage,
    ]

# Compute cumulative products, stopping when 0 is hit
cumulative = []
product = 1
for b in branches:
    if b == 0:
        break
    product *= b
    cumulative.append(product)

# Get the maximum cumulative product (number of scenarios)
num_scenarios = max(cumulative) if cumulative else 1

#print("Number of scenarios:", num_scenarios)

if case != "max_out":
    
    # Create contents
    case_and_objective_content = f"""Case and Objective Summary (in-sample test)
    -----------------------------
    Excel path: {excel_path}
    year: {year}
    cluster: {cluster}
    case: {case}

    Number of branches per stage:
    - Stage 1: {num_branches_to_firstStage}
    - Stage 2: {num_branches_to_secondStage}
    - Stage 3: {num_branches_to_thirdStage}
    - Stage 4: {num_branches_to_fourthStage}
    - Stage 5: {num_branches_to_fifthStage}
    - Stage 6: {num_branches_to_sixthStage}
    - Stage 7: {num_branches_to_seventhStage}
    - Stage 8: {num_branches_to_eighthStage}
    - Stage 9: {num_branches_to_ninthStage}
    - Stage 10: {num_branches_to_tenthStage}
    - Stage 11: {num_branches_to_eleventhStage}
    - Stage 12: {num_branches_to_twelfthStage}
    - Stage 13: {num_branches_to_thirteenthStage}
    - Stage 14: {num_branches_to_fourteenthStage}
    - Stage 15: {num_branches_to_fifteenthStage}

    Number of Scenarios: {num_scenarios}
    Number of Nodes: {num_Nodes}
    Objective Value: {objective_value:.2f}
    Investment Cost (tech+bat): {investment_cost:.2f}
    --------------------------------------------------------------------------
    SCALED TO YEARLY COSTS:
    --------------------------------------------------------------------------
    Objective Value (scaled to yearly cost): {objective_scaled_to_year:.2f}
    Investment Cost (scaled to yearly cost): {investment_cost_scaled_to_year:.2f}

    
    ---------------- COST COMPONENT BREAKDOWN: --------------------------------
    Component                        Value (EUR)     Contribution (% of Obj.)
    ---------------------------------------------------------------------------
    Revenue - Capacity Market      {Revenue_capacity_market:>15,.2f}      {Revenue_capacity_market / objective_value * 100:>8.2f}%
    Revenue - Activation Market    {Revenue_activation_market:>15,.2f}      {Revenue_activation_market / objective_value * 100:>8.2f}%
    Cost    - Day-Ahead Market     {Cost_DA_market:>15,.2f}      {Cost_DA_market / objective_value * 100:>8.2f}%
    Cost    - Intraday Market      {Cost_ID_market:>15,.2f}      {Cost_ID_market / objective_value * 100:>8.2f}%
    OPEX                            {OPEX_cost:>15,.2f}      {OPEX_cost / objective_value * 100:>8.2f}%
    Grid Tariff                     {GridTariff_cost:>15,.2f}      {GridTariff_cost / objective_value * 100:>8.2f}%
    Load Shedding                   {loadShedding_cost:>15,.2f}      {loadShedding_cost / objective_value * 100:>8.2f}%
    Investment                      {investment_cost:>15,.2f}      {investment_cost / objective_value * 100:>8.2f}%
    ---------------------------------------------------------------------------

    ---------------------- DIV ---------------------------------
    Imbalance cost related to Real-time adjustment import: {Imbalance_cost_import:.2f}
    Imbalance cost related to Real-time adjustment export: {Imbalance_cost_export:.2f}
    Total DummyFuel used by Dummy Grid: {DummyFuel_utilization:,.2f}

    Total supplied energy by gas boiler: {gasboil_utilization:,.2f}
    Total supplied energy by gas boiler with CCS: {gasboil_ccs_utilization:,.2f}
    Total supplied energy by electric boiler: {elboil_utilization:,.2f}
    Total supplied energy by heat pump: {HP_utilization:,.2f}

    ---------------------- In Persentages -----------------
    """


else: 
    # Create contents
    case_and_objective_content = f"""Case and Objective Summary (Out-of-Sample test)
    -----------------------------
    Excel path: {excel_path}
    year: {year}
    cluster: {cluster}
    case: {case}

    Number of branches per stage:
    - Stage 1: {num_branches_to_firstStage}
    - Stage 2: {num_branches_to_secondStage}
    - Stage 3: {num_branches_to_thirdStage}
    - Stage 4: {num_branches_to_fourthStage}
    - Stage 5: {num_branches_to_fifthStage}
    - Stage 6: {num_branches_to_sixthStage}
    - Stage 7: {num_branches_to_seventhStage}
    - Stage 8: {num_branches_to_eighthStage}
    - Stage 9: {num_branches_to_ninthStage}
    - Stage 10: {num_branches_to_tenthStage}
    - Stage 11: {num_branches_to_eleventhStage}
    - Stage 12: {num_branches_to_twelfthStage}
    - Stage 13: {num_branches_to_thirteenthStage}
    - Stage 14: {num_branches_to_fourteenthStage}
    - Stage 15: {num_branches_to_fifteenthStage}

    Number of Scenarios: {num_scenarios}
    Number of Nodes: {num_Nodes}
    Objective Value: {objective_value:.2f}
    Objective Value incl. investment cost:{objective_value + investment_cost_for_out_of_sample:.2f}
    Costs related to load shedding: {loadShedding_cost:.2f}
    ----------------------------------------------------
    SCALED TO YEARLY COSTS:
    ----------------------------------------------------
    Objective Value (scaled to yearly cost): {objective_scaled_to_year:.2f}
    Objective value incl. investment cost (scaled to yearly cost):{objective_scaled_to_year + investment_cost_scaled_to_year_for_out_of_sample:.2f}
    Load shedding cost (scaled to yearly cost): {loadShedding_cost_scaled_to_year:.2f}

    
    ---------------- COST COMPONENT BREAKDOWN INCL. INV-COST FROM IN-SAMPLE: -
    Component                        Value (EUR)     Contribution (% of Obj.)
    --------------------------------------------------------------------------
    Revenue - Capacity Market      {Revenue_capacity_market:>15,.2f}      {Revenue_capacity_market / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    Revenue - Activation Market    {Revenue_activation_market:>15,.2f}      {Revenue_activation_market / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    Cost    - Day-Ahead Market     {Cost_DA_market:>15,.2f}      {Cost_DA_market / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    Cost    - Intraday Market      {Cost_ID_market:>15,.2f}      {Cost_ID_market / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    OPEX                            {OPEX_cost:>15,.2f}      {OPEX_cost / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    Grid Tariff                     {GridTariff_cost:>15,.2f}      {GridTariff_cost / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    Load Shedding                   {loadShedding_cost:>15,.2f}      {loadShedding_cost / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    Investment                      {investment_cost_for_out_of_sample:>15,.2f}      {investment_cost_for_out_of_sample / (objective_value + investment_cost_for_out_of_sample) * 100:>8.2f}%
    -----------------------------------------------------------------------------

    ---------------------- DIV ---------------------------------
    Imbalance cost related to Real-time adjustment import: {Imbalance_cost_import:.2f}
    Imbalance cost related to Real-time adjustment export: {Imbalance_cost_export:.2f}
    Total DummyFuel used by Dummy Grid: {DummyFuel_utilization:,.2f}
    """

case_and_objective_path = os.path.join(result_folder, f"case_and_objective_info_{'out' if case == 'max_out' else 'in'}.txt")
with open(case_and_objective_path, "w") as f:
    f.write(case_and_objective_content)

print("Working directory:", os.getcwd())
print("Results folder will be:", result_folder)
#print("Input folder will be:", input_data_folder)




def write_updated_initial_parameters(model_instance, folder_path):
    import pandas as pd
    from pyomo.environ import value
    import os

    # === Default capacities for predefined grid technologies ===
    default_capacities = {
        "Power_Grid": 2000,
        "Biogas_Grid": 1000,
        "CH4_Grid": 1000,
        "H2_Grid": 1000,
        "Dummy_Grid": 1000,
    }

    # === Par_InitialCapacityInstalled.tab ===
    new_tech_data = []
    for tech in model_instance.Technology:
        v_new = value(model_instance.v_new_tech[tech])
        base = default_capacities.get(tech, 0)
        cap = round(v_new, 6) if v_new >= 0.01 else 0.0
        total_cap = cap + base
        new_tech_data.append({
            "Technology": tech,
            "Initial_Installed_Capacity": total_cap
        })

    df_tech = pd.DataFrame(new_tech_data)
    tech_file = os.path.join(folder_path, "Par_InitialCapacityInstalled.tab")
    df_tech.to_csv(tech_file, sep="\t", index=False)
    print(f"✅ Updated: {tech_file}")

    # === Par_MaxStorageCapacity.tab ===
    new_bat_data = [
        {
            "FlexibleLoad": bat,
            "MaxStorageCapacity": round(value(model_instance.v_new_bat[bat]), 6) 
            if value(model_instance.v_new_bat[bat]) >= 0.01 else 0.0
        }
        for bat in model_instance.FlexibleLoad
    ]
    df_bat = pd.DataFrame(new_bat_data)
    bat_file = os.path.join(folder_path, "Par_MaxStorageCapacity.tab")
    df_bat.to_csv(bat_file, sep="\t", index=False)
    print(f"✅ Updated: {bat_file}")

    # === Par_Max_ChargeDischargeRate.tab ===
    ratio_file = "Par_Power2Energy_ratio.tab"
    if not os.path.exists(ratio_file):
        raise FileNotFoundError(f"❌ Missing required ratio file: {ratio_file}")

    df_ratio = pd.read_csv(ratio_file, sep="\t")
    df_ratio.columns = df_ratio.columns.str.strip()
    df_ratio.set_index("FlexibleLoads", inplace=True)

    df_bat.set_index("FlexibleLoad", inplace=True)
    df_combined = df_bat.join(df_ratio, how="left")
    df_combined["Max_charge_discharge_rate"] = (
        df_combined["MaxStorageCapacity"] * df_combined["Power2Energy"]
    ).fillna(0.0)

    df_output = df_combined["Max_charge_discharge_rate"].reset_index()
    max_rate_file = os.path.join(folder_path, "Par_MaxChargeDischargeRate.tab")
    df_output.to_csv(max_rate_file, sep="\t", index=False)
    print(f"✅ Updated: {max_rate_file}")



#out_of_sample_folder = "Out_of_sample_results"
#write_updated_initial_parameters(our_model, result_folder)

if case in ["wide_small", "wide_medium", "wide_large", "deep_small", "deep_medium", "deep_large", "balanced_small", "balanced_medium", "max_in"]:#, "git_push"]:
    print("\n➡️  Running out-of-sample test for 'max_out' case...\n")
    
    # Determine the global out-of-sample folder based on industry
    if "Pulp" in excel_path:
        global_out_sample = os.path.join(base_dir, "Out_of_sample_pulp")
        industry_flag = "pulp"
    elif "Aluminum" in excel_path:
        global_out_sample = os.path.join(base_dir, "Out_of_sample_alu")
        industry_flag = "alu"
    else:
        raise ValueError("Unknown industry in excel_path.")
    
    # Create a local copy of the out-of-sample folder in the current run's result folder.
    local_out_sample = os.path.join(result_folder, f"Out_of_sample_{filenumber}")
    # Remove the destination folder if it already exists to make a clean copy.
    if os.path.isdir(local_out_sample):
        shutil.rmtree(local_out_sample)
    shutil.copytree(global_out_sample, local_out_sample)
    
    # Update the parameter files inside the local copy.
    write_updated_initial_parameters(our_model, local_out_sample)
    
    # Run out-of-sample test using the local copy folder, not the global one.
    import subprocess
    main_abs = os.path.join(base_dir, "main.py")
    subprocess.run(
        ["python", main_abs, "--year", str(year), "--case", "max_out", "--cluster", "season", "--industry", industry_flag, "--file", filenumber],
        cwd=local_out_sample)
    


###############################################################################################################
#################### SLETTER IN-SAMPLE FOLDER OG SPARER INVESTERING I OUT-OF-SAMPLE ###########################
################################################################################################################

if case == "max_out":
    # Delete the entire In_sample_data_{filenumber} folder
    in_sample_folder = os.path.join(result_folder, f"In_sample_data_{filenumber}")
    if os.path.isdir(in_sample_folder):
        shutil.rmtree(in_sample_folder)
        print(f"Deleted folder: {in_sample_folder}")

    files_to_save = {
    "Par_InitialCapacityInstalled.tab",
    "Par_MaxChargeDischargeRate.tab",
    "Par_MaxStorageCapacity.tab"
    }
    # In the Out_of_sample_{filenumber} folder, delete all files except those containing "tree" in their filename.
    out_of_sample_folder = os.path.join(result_folder, f"Out_of_sample_{filenumber}")
    if os.path.isdir(out_of_sample_folder):
        for fname in os.listdir(out_of_sample_folder):
            # if the filename (lowercase) does not include "tree", delete it
            if fname not in files_to_save:
                file_path = os.path.join(out_of_sample_folder, fname)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
"""
PLOT RESULTS
"""
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

def generate_unique_colors(n):
    cmap = plt.get_cmap("tab10")  # Use the tab10 colormap for distinct colors
    return [cmap(i % 10) for i in range(n)]

def plot_results_from_excel(input_file, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Construct Nodes mapping dynamically
    Nodes_mapping = {n: f"Node {n}" for n in model.Nodes}

    ########################################################################################
    ############## ENDRE FOR Å DEFINERE HVILKE VARIABLER SOM IKKE SKAL PLOTTES #############
    ########################################################################################
    exclude_sheets = ["y_max", "y_activity", "Up_shift", "Dwn_Shift", "d_flex", "I_OPEX", "I_DA", "I_ID", "I_activation", "I_cap_bid", "I_inv"]
    exclude_sheets = [x.strip().lower() for x in exclude_sheets]  # Normalize sheet names

    # Read the Excel file
    excel_file = pd.ExcelFile(input_file)

    for sheet_name in excel_file.sheet_names:
        if sheet_name.strip().lower() in exclude_sheets:
            print(f"Skipping variable: {sheet_name}") 
            continue  

        df = pd.read_excel(excel_file, sheet_name=sheet_name)

        if sheet_name in ["x_aFRR_DWN", "x_aFRR_UP"]:
            # Plot Index_1 vs second column for these sheets
            x_axis = df["Index_1"]
            y_axis = df.iloc[:, 1]  # Second column

            plt.figure(figsize=(12, 8))
            plt.plot(x_axis, y_axis, label=sheet_name, marker='o', color='blue')

            plt.title(f"{sheet_name}")
            plt.xlabel("Hours")
            plt.ylabel("Values")
            plt.legend(loc='best')
            plt.grid(True)

            plot_filename = f"{sheet_name}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, plot_filename))
            plt.close()

        elif sheet_name in ["x_aFRR_DWN_ind", "x_aFRR_UP_ind"]:
            # Handle indexed reserve market data
            if "Index_1" in df.columns and "Index_2" in df.columns:
                plt.figure(figsize=(12, 8))

                x_axis = df["Index_1"]
                value_column = df.columns[-1]
                unique_variables = df["Index_2"].dropna().unique()  # Drop NaN values
                colors = generate_unique_colors(len(unique_variables))

                for variable, color in zip(unique_variables, colors):
                    variable_data = df[df["Index_2"] == variable]
                    plt.plot(
                        variable_data["Index_1"], variable_data[value_column],
                        label=variable, marker='o', color=color
                    )

                plt.title(f"{sheet_name}")
                plt.xlabel("Hours")
                plt.ylabel("Values")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, title="Variables", borderaxespad=0.)
                plt.grid(True)

                plot_filename = f"{sheet_name}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, plot_filename))
                plt.close()

        else:
            # General plotting for other sheets
            if "Index_1" in df.columns and "Index_2" in df.columns:
                unique_index_1 = df["Index_1"].unique()

                for index_1_value in unique_index_1:
                    filtered_df = df[df["Index_1"] == index_1_value]

                    plt.figure(figsize=(12, 8))

                    if "Index_3" in filtered_df.columns:
                        variable_column = "Index_3"
                        value_column = df.columns[-1]
                        unique_variables = filtered_df[variable_column].dropna().unique()
                        colors = generate_unique_colors(len(unique_variables))

                        for variable, color in zip(unique_variables, colors):
                            variable_data = filtered_df[filtered_df[variable_column] == variable]
                            plt.plot(
                                variable_data["Index_2"], variable_data[value_column],
                                label=variable, marker='o', color=color
                            )

                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, title="Variables", borderaxespad=0.)
                    else:
                        value_column = df.columns[-1]
                        plt.plot(filtered_df["Index_2"], filtered_df[value_column], label=value_column, marker='o', color='blue')

                    # Use Nodes mapping for title
                    Nodes_name = Nodes_mapping.get(index_1_value, f"Index_1 = {index_1_value}")
                    plt.title(f"{sheet_name} ({Nodes_name})")
                    plt.xlabel("Hours")
                    plt.ylabel("Values")
                    plt.grid(True)

                    plot_filename = f"{sheet_name}_{Nodes_name.replace(' ', '_')}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, plot_filename))
                    plt.close()


# Usage
if __name__ == "__main__":
    input_excel_file = "Variable_Results.xlsx"  # Path to the Excel file
    output_plots_folder = "plots"  # Folder to save the plots

    # Generate plots
    plot_results_from_excel(input_excel_file, output_plots_folder, our_model)


def extract_demand_and_flex_demand(model):
    demand_data = []
    flex_demand_data = []

    for n in model.Nodes_RT:
        for t in model.Time:
            for e in model.EnergyCarrier:
                if e == "Electricity":
                    demand_value = pyo.value(model.Demand[n, t, e])
                    flex_demand_value = pyo.value(model.d_flex[n, t, e])

                    demand_data.append({'Nodes': n, 'Time': t, 'EnergyCarrier': e, 'Reference_Demand': demand_value})
                    flex_demand_data.append({'Nodes': n, 'Time': t, 'EnergyCarrier': e, 'flex_demand': flex_demand_value})

    # Convert to DataFrame
    demand_df = pd.DataFrame(demand_data)
    flex_demand_df = pd.DataFrame(flex_demand_data)

    return demand_df, flex_demand_df
 # Get the data
demand_df, flex_demand_df = extract_demand_and_flex_demand(our_model)

# Merge the DataFrames for unified plotting
merged_df = pd.merge(demand_df, flex_demand_df, on=['Nodes', 'Time', 'EnergyCarrier'])

# Endre denne for å plotte utvalgte noder (eks. første 4 i driftsnodene)
subset_nodes = merged_df["Nodes"].unique()[:4]
subset_df = merged_df[merged_df["Nodes"].isin(subset_nodes)]

# Plotting
plt.figure(figsize=(12, 6))

#####################################################################################
########################### FOR Å PLOTTE ALLE NODENE ################################
#####################################################################################

#for Nodes in merged_df['Nodes'].unique():
#    Nodes_data = merged_df[merged_df['Nodes'] == Nodes]
#    plt.step(Nodes_data['Time'], Nodes_data['Reference_Demand'],label=f'Demand - Nodes {Nodes}')
#    plt.step(Nodes_data['Time'], Nodes_data['flex_demand'], "--", label=f'Flex Demand - Nodes {Nodes}')


#####################################################################################
########################### FOR Å PLOTTE UTVALGTE NODER #############################
#####################################################################################
for node in subset_nodes:
    node_data = subset_df[subset_df["Nodes"] == node]
    plt.step(node_data["Time"], node_data["Reference_Demand"], label=f"Ref Demand - Node {node}", linestyle="-")
    plt.step(node_data["Time"], node_data["flex_demand"], label=f"Flex Demand - Node {node}", linestyle="--")


plt.xlabel('Time')
plt.ylabel('Demand (MW)')
plt.title('Demand and Flexible Demand Over Time')
plt.legend()
plt.grid(True)
plt.show()
"""