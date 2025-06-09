import pandas as pd
import numpy as np
import random
import os

def run_everything(excel_path, result_folder, filenumber, instance, year, cluster, num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage, num_branches_to_eleventhStage, num_branches_to_twelfthStage, num_branches_to_thirteenthStage, num_branches_to_fourteenthStage, num_branches_to_fifteenthStage):
    

    num_timesteps = 24
    num_nodes = (
        num_branches_to_firstStage + num_branches_to_firstStage*num_branches_to_secondStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage 
        + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage 
        + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage 
        + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage*num_branches_to_thirteenthStage 
        + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage*num_branches_to_thirteenthStage*num_branches_to_fourteenthStage
        + num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage*num_branches_to_thirteenthStage*num_branches_to_fourteenthStage*num_branches_to_fifteenthStage
    )
    num_firstStageNodes = num_branches_to_firstStage
    num_nodesInlastStage = max(num_branches_to_firstStage, num_branches_to_firstStage*num_branches_to_secondStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage*num_branches_to_thirteenthStage, num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage*num_branches_to_thirteenthStage*num_branches_to_fourteenthStage,num_branches_to_firstStage*num_branches_to_secondStage*num_branches_to_thirdStage*num_branches_to_fourthStage*num_branches_to_fifthStage*num_branches_to_sixthStage*num_branches_to_seventhStage*num_branches_to_eighthStage*num_branches_to_ninthStage*num_branches_to_tenthStage*num_branches_to_eleventhStage*num_branches_to_twelfthStage*num_branches_to_thirteenthStage*num_branches_to_fourteenthStage*num_branches_to_fifteenthStage)


    technologies = ["Power_Grid", "ElectricBoiler", "HP_LT", "HP_MT", "PV", "P2G", "G2P", "GasBoiler", "GasBoiler_CCS", "CHP", "CHP_CCS", "Biogas_Grid", "CH4_Grid", "CH4_H2_Mixer", "DieselReserveGenerator", "H2_Grid", "Dummy_Grid"]
    energy_carriers = ["Electricity", "LT", "MT", "H2", "CH4", "Biogas", "CH4_H2_Mix", "DummyFuel"]
    StorageTech = ["BESS_Li_Ion_1", "BESS_Redox_1", "CEAS_1", "Flywheel_1", "Hot_Water_Tank_LT_1", "H2_Storage_1", "CH4_Storage_1"]

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
    "Biogas_Grid": {1: 64.5, 2: 64.5}, #1 = Import, 2 = Export
    "CH4_Grid": {1: 39.479, 2: 39.479}, #1 = Import, 2 = Export
    "CH4_H2_Mixer": {1: 0},
    "DieselReserveGenerator": {1: 148.8},
    "H2_Grid": {1: 150.1502, 2: 150.1502}, #1 = Import, 2 = Export
    "Dummy_Grid": {1: 0} #1 = Export
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))

    def get_excel_path(excel_path):
        if "Pulp" in excel_path:
            return "Input_data_With_dummyGrid_and_RT_pulp.xlsx"
        elif "Alu" in excel_path:
            return "Input_data_With_dummyGrid_and_RT_alu.xlsx"
        else:
            raise ValueError("Unknown excel file type. Please provide a valid path containing 'pulp' or 'alu'.")
    
    excel_filename = get_excel_path(excel_path)
    excel_path_input = os.path.join(base_dir, excel_filename)

    in_sample_data_folder = os.path.join(result_folder, f"In_sample_data_{filenumber}")
    os.makedirs(in_sample_data_folder, exist_ok=True)
    
    def read_all_sheets(excel_path_input = excel_path_input, output_folder=in_sample_data_folder):
        xlsx = pd.ExcelFile(excel_path_input)
        for sheet in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=2)
            df = df.dropna(how='all')
            df.columns = df.columns.str.replace(' ', '_')
            df = df.fillna('').applymap(str)

            # build the full path into your in-sample folder
            out_file = os.path.join(output_folder, f"{sheet}.tab")
            df.to_csv(
                out_file,
                sep='\t',
                index=False,
                header=True,
                lineterminator='\n'
            )
            print(f"Saved {out_file}")

    # Call the function with your Excel file
    read_all_sheets()


    def make_tab_file(filename, data_generator, chunk_size=10_000_000, in_sample_data_folder=in_sample_data_folder):
        """
        Writes a large dataset to a .tab file in chunks using tab as a delimiter.

        Parameters:
        output_folder (str): Full path to the folder where the file should be saved.
        filename      (str): Name of the file (e.g. 'Par_ActivityCost.tab').
        data_generator:   Generator yielding pandas.DataFrame chunks.
        """
        # 1) Ensure the folder exists
        os.makedirs(in_sample_data_folder, exist_ok=True)

        # 2) Build the full file path
        filepath = os.path.join(in_sample_data_folder, filename)

        # 3) Write in chunks
        first_chunk = True
        with open(filepath, "w", newline="") as f:
            for df in data_generator:
                df.to_csv(
                    f,
                    sep="\t",
                    index=False,
                    header=first_chunk,
                    lineterminator="\n"
                )
                first_chunk = False

        print(f"Saved {filepath}")




    #####################################################################################
    ########################### SET GENERATION FUNCTIONS ################################
    #####################################################################################
    def generate_Set_TimeSteps(num_timesteps, filename = "Set_of_TimeSteps.tab"):
        def data_generator(chunk_size=10_000_000):
            # Create a DataFrame with a single column "Time" containing time steps 1 to num_timesteps.
            df = pd.DataFrame({"Time": range(1, num_timesteps + 1)})
            yield df

        make_tab_file(filename, data_generator())


    def generate_Set_of_Nodes(num_nodes, filename = "Set_of_Nodes.tab"):
        def data_generator(chunk_size=10_000_000):
            # Create a DataFrame with a single column "Node" containing node numbers 1 to num_nodes.
            df = pd.DataFrame({"Nodes": range(1, num_nodes + 1)})
            yield df

        make_tab_file(filename, data_generator())


    def generate_set_of_NodesFirst(num_branches_to_firstStage, filename = "Subset_NodesFirst.tab"):
        def data_generator(chunk_size=10_000_000):
            # Create a DataFrame with a single column "Node" containing node numbers 1 to num_nodes.
            df = pd.DataFrame({"Nodes": range(1, num_branches_to_firstStage + 1)})
            yield df

        make_tab_file(filename, data_generator())


    def generate_set_of_Parents(num_nodes, filename = "Set_of_Parents.tab"):
        def data_generator(chunk_size=10_000_000):
            df = pd.DataFrame({"Parent": range(1, num_nodes - num_nodesInlastStage + 1)})
            yield df
        
        make_tab_file(filename, data_generator())


    def generate_set_Parent_Coupling(list_of_branches, filename = "Set_ParentCoupling.tab"):
        parent_mapping = []  # To store rows with "Node" and "Parent"
        
        # Stage 1: The root nodes (these do not appear in the output, as they have no parent)
        current_stage = list(range(1, list_of_branches[0] + 1))
        node_counter = current_stage[-1]  # Last node number in stage 1

        # For each subsequent stage, generate children for each node in the current stage.
        # The branch_counts list has one entry per stage; stage 1 is already defined.
        for stage_index in range(1, len(list_of_branches)):
            next_stage = []
            branches = list_of_branches[stage_index]  # Number of children per parent for this stage
            for parent in current_stage:
                for _ in range(branches):
                    node_counter += 1
                    child = node_counter
                    parent_mapping.append({"Node": child, "Parent": parent})
                    next_stage.append(child)
            current_stage = next_stage  # Update for the next stage

        def data_generator(chunk_size=10_000_000):
            # In this example, the entire mapping is yielded as one chunk.
            yield pd.DataFrame(parent_mapping)
        
        make_tab_file(filename, data_generator())

    ###############################################################################
    ################# DENNE MÅ LAGES TIDLIGERE ENN DE ANDRE #######################
    ###############################################################################
    generate_set_Parent_Coupling([num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage, num_branches_to_eleventhStage, num_branches_to_twelfthStage, num_branches_to_thirteenthStage, num_branches_to_fourteenthStage, num_branches_to_fifteenthStage])

    def generate_set_of_NodesInStage(branch_counts, filename = "Set_of_NodesInStage.tab"):
        nodes_in_stage = []  # To hold rows of the form {"Nodes": child_node, "Period": period}
        
        # Define the root nodes (stage 1) – these are not output since they have no parent period.
        current_stage = list(range(1, branch_counts[0] + 1))
        node_counter = current_stage[-1]  # Last node number in stage 1
        
        # For each subsequent stage, generate children and record their period (stage_index).
        # Here, stage 2 corresponds to Period 1, stage 3 to Period 2, etc.
        for stage_index in range(1, len(branch_counts)):
            next_stage = []
            period = stage_index  # period = stage_index (so stage 2 -> period 1, stage 3 -> period 2, etc.)
            for parent in current_stage:
                for _ in range(branch_counts[stage_index]):
                    node_counter += 1
                    child = node_counter
                    next_stage.append(child)
                    nodes_in_stage.append({"Nodes": child, "Period": period})
            current_stage = next_stage

        def data_generator(chunk_size=10_000_000):
            # Yield the entire mapping as one chunk.
            yield pd.DataFrame(nodes_in_stage)
        
        make_tab_file(filename, data_generator())


    def generate_set_of_Periods(branch_counts, filename = "Set_of_Periods.tab"):
        def data_generator(chunk_size=10_000_000):
            # Consider stages 2 and beyond (i.e. branch_counts[1:]) and count only those > 0.
            valid_periods = [i for i, count in enumerate(branch_counts[1:], start=1) if count != 0]
            # Create a DataFrame with valid period numbers.
            df = pd.DataFrame({"Periods": valid_periods})
            yield df

        make_tab_file(filename, data_generator())


    def generate_node_probability(branch_counts):
        node_prob = {}
        current_node = 1

        # Process stage 1:
        stage1_nodes = branch_counts[0]
        if stage1_nodes == 0:
            raise ValueError("The first stage must have at least one node.")
        prob = 1.0 / stage1_nodes
        for _ in range(stage1_nodes):
            node_prob[current_node] = prob
            current_node += 1

        cumulative = stage1_nodes
        # For each subsequent stage:
        for i in range(1, len(branch_counts)):
            if branch_counts[i] == 0:
                # Skip this stage if branch count is zero.
                continue
            stage_nodes = cumulative * branch_counts[i]  # number of nodes in this stage
            prob = 1.0 / stage_nodes
            for _ in range(stage_nodes):
                node_prob[current_node] = prob
                current_node += 1
            cumulative *= branch_counts[i]
        return node_prob




    NodeProbability = generate_node_probability([num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage, num_branches_to_eleventhStage, num_branches_to_twelfthStage, num_branches_to_thirteenthStage, num_branches_to_fourteenthStage, num_branches_to_fifteenthStage])

    ###############################################################################
    # ----------------- Parameter data  -----------------
    ###############################################################################

    # --------------------------
    # Dictionaries
    # --------------------------
    if instance == 1: #Expected value
        if year == 2025:
            CostExpansion_Tec = {
                "Power_Grid": 1_000_000,
                "ElectricBoiler": 27.95,
                "HP_LT": 250.126,
                "HP_MT": 299.647,
                "PV": 213.391,
                "P2G": 222.977,
                "G2P": 719.551,
                "GasBoiler": 19.289,
                "GasBoiler_CCS": 230.209,
                "CHP": 277.527,
                "CHP_CCS": 488.447,
                "Biogas_Grid": 1_000_000,
                "CH4_Grid": 1_000_000,
                "CH4_H2_Mixer": 0,
                "DieselReserveGenerator": 125.536,
                "H2_Grid": 1_000_000,
                "Dummy_Grid": 1_000_000
            }
            

            CostExpansion_Bat = {
                "BESS_Li_Ion_1": 270.257,
                "BESS_Redox_1": 154.252,
                "CAES_1": 224.152,
                "Flywheel_1": 128.768,
                "Hot_Water_Tank_LT_1": 0.797,
                "H2_Storage_1": 15.124,
                "CH4_Storage_1": 0.03239
            }
        elif year == 2050:
            CostExpansion_Tec = {
                "Power_Grid": 1_000_000,
                "ElectricBoiler": 25.92968,
                "HP_LT": 212.06173,
                "HP_MT": 254.07956,
                "PV": 126.85121,
                "P2G": 94.45389,
                "G2P": 479.70075,
                "GasBoiler": 17.33577,
                "GasBoiler_CCS": 222.95651,
                "CHP": 270.55360,
                "CHP_CCS": 476.17433,
                "Biogas_Grid": 1_000_000,
                "CH4_Grid": 1_000_000,
                "CH4_H2_Mixer": 0,
                "DieselReserveGenerator": 123.99813,
                "H2_Grid": 1_000_000,
                "Dummy_Grid": 1_000_000
            }
            
            CostExpansion_Bat = {
                "BESS_Li_Ion_1": 86.10438,
                "BESS_Redox_1": 106.90110,
                "CAES_1": 217.38136,
                "Flywheel_1": 128.76774,
                "Hot_Water_Tank_LT_1": 0.79737,
                "H2_Storage_1": 6.23917,
                "CH4_Storage_1": 0.03239
            }

        else:
            raise ValueError("Invalid year. Please choose either 2025 or 2050.")

    elif instance == 2 or instance == 5: #Lowerbound
        if year == 2025:
            CostExpansion_Tec = {
                "Power_Grid": 1_000_000,
                "ElectricBoiler": 22.36,
                "HP_LT": 200.1008,
                "HP_MT": 239.7176,
                "PV": 170.7128,
                "P2G": 178.3816,
                "G2P": 575.6408,
                "GasBoiler": 15.4312,
                "GasBoiler_CCS": 184.1672,
                "CHP": 222.0216,
                "CHP_CCS": 390.7576,
                "Biogas_Grid": 1_000_000,
                "CH4_Grid": 1_000_000,
                "CH4_H2_Mixer": 0,
                "DieselReserveGenerator": 100.4288,
                "H2_Grid": 1_000_000,
                "Dummy_Grid": 1_000_000
            }

            CostExpansion_Bat = {
                "BESS_Li_Ion_1": 216.2056,
                "BESS_Redox_1": 123.4016,
                "CAES_1": 179.3216,
                "Flywheel_1": 103.0144,
                "Hot_Water_Tank_LT_1": 0.6376,
                "H2_Storage_1": 12.0992,
                "CH4_Storage_1": 0.025912
            }

        elif year == 2050:
            CostExpansion_Tec = {
                "Power_Grid": 1_000_000,
                "ElectricBoiler": 20.74374,
                "HP_LT": 169.64938,
                "HP_MT": 203.26365,
                "PV": 101.48097,
                "P2G": 75.56311,
                "G2P": 383.7606,
                "GasBoiler": 13.86862,
                "GasBoiler_CCS": 178.36521,
                "CHP": 216.44288,
                "CHP_CCS": 380.93946,
                "Biogas_Grid": 1_000_000,
                "CH4_Grid": 1_000_000,
                "CH4_H2_Mixer": 0,
                "DieselReserveGenerator": 99.198504,
                "H2_Grid": 1_000_000,
                "Dummy_Grid": 1_000_000
            }

            CostExpansion_Bat = {
                "BESS_Li_Ion_1": 68.883504,
                "BESS_Redox_1": 85.52088,
                "CAES_1": 173.905088,
                "Flywheel_1": 103.014192,
                "Hot_Water_Tank_LT_1": 0.637896,
                "H2_Storage_1": 4.991336,
                "CH4_Storage_1": 0.025912
            }


        else:
            raise ValueError("Invalid year. Please choose either 2025 or 2050.")
        
    elif instance == 3 or instance == 4: #Upperbound
        if year == 2025:
            CostExpansion_Tec = {
                "Power_Grid": 1_000_000,
                "ElectricBoiler": 33.54,
                "HP_LT": 300.1512,
                "HP_MT": 359.5764,
                "PV": 256.0692,
                "P2G": 267.5724,
                "G2P": 863.4612,
                "GasBoiler": 23.1468,
                "GasBoiler_CCS": 276.2508,
                "CHP": 333.0324,
                "CHP_CCS": 586.1364,
                "Biogas_Grid": 1_000_000,
                "CH4_Grid": 1_000_000,
                "CH4_H2_Mixer": 0,
                "DieselReserveGenerator": 150.6432,
                "H2_Grid": 1_000_000,
                "Dummy_Grid": 1_000_000
            }

            CostExpansion_Bat = {
                "BESS_Li_Ion_1": 324.3084,
                "BESS_Redox_1": 185.1024,
                "CAES_1": 268.9824,
                "Flywheel_1": 154.5216,
                "Hot_Water_Tank_LT_1": 0.9564,
                "H2_Storage_1": 18.1488,
                "CH4_Storage_1": 0.038868
            }

        elif year == 2050:
            CostExpansion_Tec = {
                "Power_Grid": 1_000_000,
                "ElectricBoiler": 31.115616,
                "HP_LT": 254.474076,
                "HP_MT": 304.895472,
                "PV": 152.221452,
                "P2G": 113.344668,
                "G2P": 575.6409,
                "GasBoiler": 20.802924,
                "GasBoiler_CCS": 267.547812,
                "CHP": 324.66432,
                "CHP_CCS": 571.409196,
                "Biogas_Grid": 1_000_000,
                "CH4_Grid": 1_000_000,
                "CH4_H2_Mixer": 0,
                "DieselReserveGenerator": 148.797756,
                "H2_Grid": 1_000_000,
                "Dummy_Grid": 1_000_000
            }

            CostExpansion_Bat = {
                "BESS_Li_Ion_1": 103.325256,
                "BESS_Redox_1": 128.28132,
                "CAES_1": 260.857632,
                "Flywheel_1": 154.521288,
                "Hot_Water_Tank_LT_1": 0.956844,
                "H2_Storage_1": 7.487004,
                "CH4_Storage_1": 0.038868
            }



        else:
            raise ValueError("Invalid year. Please choose either 2025 or 2050.")
    else:
        raise ValueError("Invalid instance number.")
    # --------------------------    --------------------------
    # --------------------------    --------------------------      

    CostGridTariff = 123.93




    ####################################################################################
    ########################### GET CHILD MAPPINNG FUNC #################################
    #####################################################################################

    def map_children_to_parents_from_file(tab_filename, in_sample_data_folder = in_sample_data_folder):
        filepath = os.path.join(in_sample_data_folder, tab_filename)
        # Les tab-fila (antatt tab-separert)
        df = pd.read_csv(filepath, sep="\t")
        
        # Bygg et dictionary med umiddelbare relasjoner: barn -> forelder
        child_to_parent = {row["Node"]: row["Parent"] for _, row in df.iterrows()}
        
        # Funksjon for å finne top-level forelder ved å følge oppover i treet
        def find_top(node):
            # Hvis node ikke finnes som barn (nøkkel) i child_to_parent,
            # er den top-level (det antas at top-level foreldre kun er i Parent-kolonnen)
            if node not in child_to_parent:
                return node
            else:
                return find_top(child_to_parent[node])
        
        # Beregn top-level for alle noder (som finnes som barn)
        top_level = {}
        for node in child_to_parent:
            top_level[node] = find_top(node)
        
        # Gruppér noder etter top-level forelder
        grouping = {}
        for node, top in top_level.items():
            grouping.setdefault(top, []).append(node)
        
        return grouping

    def extract_parent_coupling(in_sample_data_folder = in_sample_data_folder, tab_filename = "Set_ParentCoupling.tab"):
        filepath = os.path.join(in_sample_data_folder, tab_filename)
        df = pd.read_csv(filepath, sep="\t")
        data = {
            "Node": df["Node"].tolist(),
            "Parent": df["Parent"].tolist()
        }
        return data

    #data = extract_parent_coupling()
    #df_example = pd.DataFrame(data)
    taB_filenam = "Set_ParentCoupling.tab"
    #df_example.to_csv(taB_filenam, sep = "\t", index=False, header=True, lineterminator='\n')
    mapping = map_children_to_parents_from_file(taB_filenam)
    print("Førstestegs-forelder : -> [alle etterkommere]:")
    mapping_converted = {int(k): [int(x) for x in v] for k, v in mapping.items()}
    print(mapping_converted)

    ####################################################################################
    ########################### GET PARENT MAPPING FUNC #################################
    #####################################################################################
    def create_parent_mapping(filename, in_sample_data_folder = in_sample_data_folder):
        """
        Leser en .tab-fil med kolonnene 'Node' og 'Parent',
        og returnerer en parent_mapping som et Python-dictionary.
        """
        filepath = os.path.join(in_sample_data_folder, filename)
        # Les filen
        df = pd.read_csv(filepath, sep="\t")
        
        # Sjekk at nødvendige kolonner finnes
        if not {"Node", "Parent"}.issubset(df.columns):
            raise ValueError("Filen må inneholde kolonnene 'Node' og 'Parent'.")

        # Lag parent_mapping
        parent_mapping = dict(zip(df["Node"], df["Parent"]))
        
        return parent_mapping
    


    def invert_parent_mapping(parent_mapping):
        inverted = {}
        for node, parent in parent_mapping.items():
            inverted.setdefault(parent, []).append(node)
        return inverted

    
    

    # ----------------- HISTORICAL PRICE DATA HANDLING -----------------

    # Load data
    df = pd.read_excel(excel_path, sheet_name="2024 NO1 data")

    # Group by full (month, day) sets
    df_grouped = df.groupby(["Month", "Day"])
    day_data_map = {
        (month, day): group.reset_index(drop=True)
        for (month, day), group in df_grouped
        if len(group) == 24
    }

    # ---- Replace the manual node_month_ranges with parent-group-based logic ----
    #####################################################################################
    ########################### CLUSTER BASERT PÅ 4 ÅRSTIDER ############################
    #####################################################################################
    """
    parent_month_mapping = {
        1: [12, 1, 2],
        2: [3, 4, 5],
        3: [6, 7, 8],
        4: [9, 10, 11],
    }
    """
    #####################################################################################
    ########################### CLUSTER BASERT PÅ 2 ÅRSTIDER ############################
    #####################################################################################

    # Extend parent_month_mapping assignment to parent nodes as well


        # === Define required days to ensure inclusion ===
    
    
 # Assume these mappings are built earlier:
#   mapping_parents = invert_parent_mapping( create_parent_mapping("Set_ParentCoupling.tab") )
#   mapping_converted = {int(k): [int(x) for x in v] for k, v in 
#                          map_children_to_parents_from_file("Set_ParentCoupling.tab").items()}

        # And parent_month_mapping is defined as (for 2 seasons):
        # --- Define allowed months for each season ---
   
    # Assume these dictionaries exist:
# mapping_parents = {1: [3, 4], 2: [5, 6], 3: [7, 8], 4: [9, 10], 5: [11, 12], 6: [13, 14]}
# mapping_seasonal = {1: [3, 4, 7, 8, 9, 10], 2: [5, 6, 11, 12, 13, 14]}
# And required_days = [(1,5), (1,8), (1,9), (1,16), (1,18), (1,19), (2,9)]
# Also day_data_map is defined and parent_month_mapping as below:

    # Example usage:
    original_mapping = create_parent_mapping("Set_ParentCoupling.tab")
    mapping_parents = invert_parent_mapping(original_mapping)
    print("mapping_parents", mapping_parents)

    mapping_seasonal = mapping_converted

    print("mapping_seasonal", mapping_seasonal)
    

    parent_month_mapping = {
        1: [4, 5, 6, 7, 8, 9],      # summer
        2: [1, 2, 3, 10, 11, 12]     # winter
    }
    required_days_pulp = [(1, 5), (1, 8), (1, 9), (1, 16), (1, 18), (1, 19), (2, 9)]
    required_days_alu =  [(1, 5), (1, 6), (1, 7), (1, 8), (1, 20)]

   
    required_days_high_diversed_pulp = [ #Above 700 peak
    # January
    (1, 4), (1, 5), (1, 8), (1, 9),
    (1, 10), (1, 12), (1, 15), (1, 16),
    (1, 18), (1, 19),

    # February
    (2, 8), (2, 9),
    ]

    required_days_high_diversed_alu = [ #Over 9
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 12),
    (1, 13),
    (1, 14),
    (1, 15),
    (1, 16),
    (1, 18),
    (1, 19),
    (1, 20),
    (2, 9),
    (2, 10)
    ]

    required_days_low_diversed_pulp = [ #Below 250 peak
    # May
    (5, 1), (5, 2), (5, 3), (5, 13), (5, 14), (5, 15),
    (5, 16), (5, 17), (5, 20), (5, 21), (5, 22), (5, 23),
    (5, 24), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31),

    # June
    (6, 3), (6, 4), (6, 5), (6, 11), (6, 12), (6, 17),
    (6, 18), (6, 19), (6, 20), (6, 21), (6, 24), (6, 25),
    (6, 26), (6, 27), (6, 28),

    # July
    (7, 1), (7, 2), (7, 3), (7, 8), (7, 9), (7, 10),
    (7, 11), (7, 12), (7, 15), (7, 16), (7, 17), (7, 18),
    (7, 19), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26),
    (7, 29), (7, 30), (7, 31),

    # August
    (8, 1), (8, 2), (8, 5), (8, 6), (8, 7), (8, 8),
    (8, 9), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16),
    (8, 19), (8, 20), (8, 21), (8, 22), (8, 23), (8, 26),
    (8, 27), (8, 28), (8, 29), (8, 30),

    # September
    (9, 3), (9, 4), (9, 5), (9, 6), (9, 9), (9, 10),
    (9, 16), (9, 18), (9, 19), (9, 23), (9, 24), (9, 25),
    ]

    required_days_low_diversed_alu = [ #Below 3
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
    (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19), (5, 20),
    (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27), (5, 28), (5, 29), (5, 30), (5, 31),
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
    (6, 11), (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (6, 20),
    (6, 21), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (6, 28), (6, 29), (6, 30),
    (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10),
    (7, 11), (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19), (7, 20),
    (7, 21), (7, 22), (7, 23), (7, 24), (7, 25), (7, 26), (7, 27), (7, 28), (7, 29), (7, 30), (7, 31),
    (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10),
    (8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19), (8, 20),
    (8, 21), (8, 22), (8, 23), (8, 24), (8, 25), (8, 26), (8, 27), (8, 28), (8, 29), (8, 30), (8, 31),
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10),
    (9, 11), (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19), (9, 20),
    (9, 21), (9, 22), (9, 23), (9, 24), (9, 25), (9, 26), (9, 27), (9, 28), (9, 29), (9, 30),
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
    (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20),
    (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 26), (10, 27), (10, 28), (10, 29), (10, 30), (10, 31),
    (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),
    (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 20), (11, 22),
    (11, 23), (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (11, 29), (11, 30),
    (12, 1), (12, 2), (12, 3), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 13), (12, 14),
    (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), (12, 21), (12, 22), (12, 23), (12, 24),
    (12, 25), (12, 26), (12, 27), (12, 28), (12, 29), (12, 30), (12, 31)
    ]

    required_days_medium_diversed_pulp = [ #Between  450 and 550 peak
    # January
    (1, 24), (1, 25), (1, 26), (1, 30), (1, 31),

    # February
    (2, 5),  (2, 14), (2, 15), (2, 16), (2, 19),
    (2, 20), (2, 21), (2, 26), (2, 27),

    # March
    (3, 4),  (3, 5),  (3, 6),  (3, 7),  (3, 8),
    (3, 11), (3, 12), (3, 18), (3, 19), (3, 20),
    (3, 25), (3, 26),

    # April
    (4, 3),  (4, 4),  (4, 5),  (4, 19),

    # November
    (11, 12), (11, 14), (11, 18), (11, 19), (11, 20),
    (11, 28), (11, 29),

    # December
    (12, 3),  (12, 5),  (12, 6),  (12, 9),  (12, 13),
    (12, 20), (12, 23), (12, 24), (12, 30), (12, 31),
    ]
    # … assume mapping_parents, mapping_seasonal, parent_month_mapping,
    #    day_data_map and required_days are already defined …

    required_days_medium_diversed_alu = [ #Between 5 and 7
    
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (1, 7), (1, 8), (1, 9), (1,10), (1,11), (1,12),
    (1,13), (1,14), (1,15), (1,16), (1,17), (1,18),
    (1,19), (1,20), (1,21), (1,24), (1,25), (1,26),
    (1,27), (1,28), (1,29), (1,30),

    (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9),
    (2,10), (2,11), (2,12), (2,13), (2,14), (2,15),
    (2,16), (2,17), (2,18), (2,19), (2,20), (2,24),
    (2,25), (2,26), (2,27),

    (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8),
    (3, 9), (3,10), (3,11), (3,16), (3,17), (3,18),
    (3,19), (3,23), (3,24), (3,25),

    (4, 3), (4, 4), (4, 5), (4, 6), (4,19), (4,20),
    (4,22),

    (11, 2), (11,10), (11,12), (11,17), (11,18), (11,19),
    (11,20), (11,21), (11,22), (11,23), (11,24), (11,28),
    (11,29),

    (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8),
    (12, 9), (12,10), (12,11), (12,12), (12,13), (12,14),
    (12,15), (12,20), (12,21), (12,22), (12,23), (12,30),
    (12,31)

    ] 

    node_to_day = {}

    

    # ─── your pre-existing data structures ────────────────────────────
    # mapping_parents  = {1:[3,4], 2:[5,6], 3:[7,8], 4:[9,10], 5:[11,12], 6:[13,14]}
    # mapping_seasonal = {1:[…summer nodes…], 2:[…winter nodes…]}
    # parent_month_mapping = {
    #     1: [4,5,6,7,8,9],      # summer
    #     2: [1,2,3,10,11,12],   # winter
    # }
    # required_days    = [(1,5),(1,8),(1,9),(1,16),(1,18),(1,19),(2,9)]
    # day_data_map     = { (m,d): pd.DataFrame with an "MT" column … }

    """

    if cluster in ["random", "season"]:
        # ── 1) ASSIGN season roots + their children ─────────────────
        for season, nodes in mapping_seasonal.items():
            allowed = parent_month_mapping.get(season, list(range(1,13)))
            valid_days = [d for d in day_data_map if d[0] in allowed]
            
            node_to_day[season] = random.choice(valid_days)
            for n in nodes:
                node_to_day[n] = random.choice(valid_days)

        # ── 2) INJECT one required_day into a random winter node ─────
        available = [d for d in required_days if d in day_data_map]
        if not available:
            print("⚠️ No required_days are in day_data_map!")
        else:
            winter_nodes   = mapping_seasonal[2]
            inj_node       = random.choice(winter_nodes)
            inj_day        = random.choice(available)
            node_to_day[inj_node] = inj_day
            print(f"⛄ [random/season] Injected required day {inj_day} into node {inj_node}")
    """
    if cluster == "random":
        if "Pulp" in excel_path:
            required_days = required_days_pulp
        elif "Alu" in excel_path:
            required_days = required_days_alu
        else:
            raise ValueError("Unknown excel_path, cannot determine required_days.")
        
        parent_month_mapping = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }

        all_nodes = set()
        for parent, children in mapping_converted.items():
            all_nodes.update([parent] + children)
        all_nodes.update(range(1, num_firstStageNodes + 1))

        max_attempts = 1000
        success = False

        for attempt in range(max_attempts):
            temp_node_to_day = {}
            failed_node = None

            try:
                for parent, child_nodes in mapping_converted.items():
                    allowed_months = parent_month_mapping.get(parent, [1, 2, 3])
                    valid_days = [d for d in day_data_map if d[0] in allowed_months]

                    if not valid_days:
                        failed_node = f"Parent group {parent}"
                        raise ValueError()

                    for node in [parent] + child_nodes:
                        temp_node_to_day[node] = random.choice(valid_days)

                for node in range(1, num_firstStageNodes + 1):
                    if node in temp_node_to_day:
                        continue
                    allowed_months = parent_month_mapping.get(node, [1, 2, 3])
                    valid_days = [d for d in day_data_map if d[0] in allowed_months]

                    if not valid_days:
                        failed_node = f"First-stage node {node}"
                        raise ValueError()

                    temp_node_to_day[node] = random.choice(valid_days)

            except ValueError:
                print(f"Attempt {attempt + 1}: No valid days found for {failed_node} (allowed months: {allowed_months})")
                continue  # try again

            included_required_days = [day for day in temp_node_to_day.values() if day in required_days]
           
            if included_required_days:
                node_to_day = temp_node_to_day
                success = True
                print(f"[Success after {attempt + 1} attempts] ✅ Required day found!")
                print("Included required day(s):", included_required_days)
                break

        if not success:
            raise RuntimeError(f"❌ Failed to sample data including any required day after {max_attempts} attempts.")
       
    elif cluster == "season":
        if "Pulp" in excel_path:
            required_days = required_days_pulp
        elif "Alu" in excel_path:
            required_days = required_days_alu
        else:
            raise ValueError("Unknown excel_path, cannot determine required_days.")
        
        parent_month_mapping = {
            1: [4, 5, 6, 7, 8, 9],       # Spring/Summer
            2: [1, 2, 3, 10, 11, 12],    # Winter
        }

        all_nodes = set()
        for parent, children in mapping_converted.items():
            all_nodes.update([parent] + children)
        all_nodes.update(range(1, num_firstStageNodes + 1))

        max_attempts = 1000
        success = False

        for attempt in range(max_attempts):
            temp_node_to_day = {}
            failed_node = None

            try:
                for parent, child_nodes in mapping_converted.items():
                    allowed_months = parent_month_mapping.get(parent, [1, 2, 3])
                    valid_days = [d for d in day_data_map if d[0] in allowed_months]

                    if not valid_days:
                        failed_node = f"Parent group {parent}"
                        raise ValueError()

                    for node in [parent] + child_nodes:
                        temp_node_to_day[node] = random.choice(valid_days)

                for node in range(1, num_firstStageNodes + 1):
                    if node in temp_node_to_day:
                        continue
                    allowed_months = parent_month_mapping.get(node, [1, 2, 3])
                    valid_days = [d for d in day_data_map if d[0] in allowed_months]

                    if not valid_days:
                        failed_node = f"First-stage node {node}"
                        raise ValueError()

                    temp_node_to_day[node] = random.choice(valid_days)

            except ValueError:
                print(f"[Attempt {attempt + 1}] ❌ No valid days found for {failed_node} (allowed months: {allowed_months})")
                continue

            # Normalize and check for required days
            included_required_days = [
                (int(day[0]), int(day[1]))
                for day in temp_node_to_day.values()
                if (int(day[0]), int(day[1])) in required_days
            ]

            if included_required_days:
                node_to_day = temp_node_to_day
                success = True
                print(f"[Success after {attempt + 1} attempts] ✅ Required day found!")
                print("Included required day(s):", included_required_days)
                print("Execution continued from: <required day inclusion check in 'season' cluster>")
                break

        if not success:
            raise RuntimeError(f"❌ Failed to sample data including any required day after {max_attempts} attempts.")


    elif cluster == "guided":
        if "Pulp" in excel_path:
            required_days = required_days_pulp
        elif "Alu" in excel_path:
            required_days = required_days_alu
        else:
            raise ValueError("Unknown excel_path, cannot determine required_days.")
        # ── A) ASSIGN season roots + their children ─────────────────
        for season, nodes in mapping_seasonal.items():
            allowed = parent_month_mapping.get(season, list(range(1,13)))
            valid_days = [d for d in day_data_map if d[0] in allowed]
            
            node_to_day[season] = random.choice(valid_days)
            for n in nodes:
                node_to_day[n] = random.choice(valid_days)

        # ── B) INJECT one required_day into a random winter node ─────
        available = [d for d in required_days if d in day_data_map]
        injected_parent = None

        if not available:
            print("⚠️ No required_days are in day_data_map!")
        else:
            winter_nodes   = mapping_seasonal[2]
            inj_node       = random.choice(winter_nodes)
            inj_day        = random.choice(available)
            node_to_day[inj_node] = inj_day
            print(f"⛄ [guided]  Injected required day {inj_day} into node {inj_node}")

            # find its parent
            injected_parent = next(
                (p for p, kids in mapping_parents.items() if inj_node in kids),
                None
            )

            if injected_parent is not None:
                # find which season that parent belongs to
                season_of_parent = next(
                    s for s,nodes in mapping_seasonal.items()
                    if injected_parent == s or injected_parent in nodes
                )
                allowed = parent_month_mapping[season_of_parent]
                valid_days = [d for d in day_data_map if d[0] in allowed]

                # compute ±10% band around the injected day's MT peak
                base_peak = day_data_map[inj_day]["MT"].max()
                lo, hi = base_peak * 0.90, base_peak * 1.10
                sibling_days = [
                    d for d in valid_days
                    if lo <= day_data_map[d]["MT"].max() <= hi
                ]

                # *** ONLY update the injected node’s siblings ***
                siblings = [
                    c for c in mapping_parents[injected_parent]
                    if c != inj_node
                ]
                for sib in siblings:
                    node_to_day[sib] = (
                        random.choice(sibling_days)
                        if sibling_days else
                        random.choice(valid_days)
                    )
                print(f"🔧 [guided] Updated siblings of parent {injected_parent}: {siblings}")

        # ── C) ENFORCE ±10% rule on all *other* parent groups ─────────
        for parent, children in mapping_parents.items():
            if parent == injected_parent or not children:
                continue

            # pick season for this group
            parent_season = next(
                (s for s,nodes in mapping_seasonal.items()
                if parent == s or any(c in nodes for c in children)),
                None
            )
            allowed = parent_month_mapping.get(parent_season, list(range(1,13)))
            valid_days = [d for d in day_data_map if d[0] in allowed]

            # first child is “base”
            base_child = children[0]
            base_day   = random.choice(valid_days)
            node_to_day[base_child] = base_day

            base_peak = day_data_map[base_day]["MT"].max()
            lo, hi = base_peak * 0.90, base_peak * 1.10
            sib_choices = [
                d for d in valid_days
                if d != base_day and lo <= day_data_map[d]["MT"].max() <= hi
            ]

            for sib in children[1:]:
                node_to_day[sib] = (
                    random.choice(sib_choices)
                    if sib_choices else base_day
                )

        print("✅ Guided clustering with ±10% MT tolerance complete.")

    
    

    elif cluster == "diversed":
        if "Pulp" in excel_path:
            required_days_high_diversed   = required_days_high_diversed_pulp
            required_days_low_diversed    = required_days_low_diversed_pulp
            required_days_medium_diversed = required_days_medium_diversed_pulp
        elif "Alu" in excel_path:
            required_days_high_diversed   = required_days_high_diversed_alu
            required_days_low_diversed    = required_days_low_diversed_alu
            required_days_medium_diversed = required_days_medium_diversed_alu
        else:
            raise ValueError("Unknown excel_path, cannot determine required_days.")

        # 1) Pre-filter required days that actually exist
        highs   = [d for d in required_days_high_diversed   if d in day_data_map]
        lows    = [d for d in required_days_low_diversed    if d in day_data_map]
        mediums = [d for d in required_days_medium_diversed if d in day_data_map]

        if not highs or not lows or not mediums:
            raise RuntimeError("⚠️  One of the required_days_high/low/medium sets is empty!")

        all_days = list(day_data_map.keys())

        # 2) For each sibling set, assign high, low, medium in turn
        for parent, children in mapping_parents.items():
            # work on a mutable copy
            remaining = children.copy()

            # HIGH
            if remaining:
                c_high = random.choice(remaining)
                day_high = random.choice(highs)
                node_to_day[c_high] = day_high
                remaining.remove(c_high)
                print(f"[diversed] Parent {parent}: Node {c_high} ← HIGH {day_high}")

            # LOW
            if remaining:
                c_low = random.choice(remaining)
                day_low = random.choice(lows)
                node_to_day[c_low] = day_low
                remaining.remove(c_low)
                print(f"[diversed] Parent {parent}: Node {c_low} ← LOW  {day_low}")

            # MEDIUM
            if remaining:
                c_med = random.choice(remaining)
                day_med = random.choice(mediums)
                node_to_day[c_med] = day_med
                remaining.remove(c_med)
                print(f"[diversed] Parent {parent}: Node {c_med} ← MED  {day_med}")

            # ANY LEFT: pure random
            for c in remaining:
                rand_day = random.choice(all_days)
                node_to_day[c] = rand_day
                print(f"[diversed] Parent {parent}: Node {c} ← RAND {rand_day}")

        # 3) Finally fill every other node (including parents and any first-stage nodes)
        all_nodes = set(mapping_parents.keys())
        for kids in mapping_parents.values():
            all_nodes.update(kids)
        all_nodes.update(range(1, num_firstStageNodes + 1))

        for node in sorted(all_nodes):
            if node not in node_to_day:
                node_to_day[node] = random.choice(all_days)

        print("✅ Diversed clustering complete: each sibling set has one HIGH, one LOW, one MEDIUM (if ≥3), rest random.")


    else:
        raise ValueError(f"Unknown cluster type: {cluster}")

    # ─── FINAL OUTPUT ────────────────────────────────────────────────
    print("\n📅 Selected day for each node:")
    for node in sorted(node_to_day):
        m, d = node_to_day[node]
        print(f"Node {node}: Month={m:02d}, Day={d:02d}")


    def extract_series_for_column(columns, node_to_day, day_data_map, all_keys=None, fill_zero_for_missing=True):
        """
        Extracts 24-hour time series data for specified columns across nodes.

        Parameters:
        - columns: list of column names in the Excel file to extract
        - node_to_day: mapping of node -> (month, day)
        - day_data_map: mapping of (month, day) -> DataFrame with hourly data
        - all_keys: list of all expected keys (e.g., all fuels or all price types)
        - fill_zero_for_missing: if True, fill missing keys with zero time series

        Returns:
        - Dictionary: {key: {node: {timestep: value}}}
        """
        result = {}

        for col in columns:
            result[col] = {}
            for node, (month, day) in node_to_day.items():
                df_day = day_data_map[(month, day)]
                result[col][node] = {t + 1: float(df_day[col].iloc[t]) for t in range(24)}

        if fill_zero_for_missing and all_keys:
            for key in all_keys:
                if key not in result:
                    result[key] = {node: {t: 0.0 for t in range(1, 25)} for node in node_to_day}

        return result

    # ✅ Define demand-related inputs
    demand_columns = ["Electricity", "LT", "MT", "CH4"]
    all_fuels = ["Electricity", "LT", "MT", "H2", "CH4", "Biogas", "CH4_H2_Mix"]

    # ✅ Build ReferenceDemand using the unified extractor
    ReferenceDemand = extract_series_for_column(
        columns=demand_columns,
        node_to_day=node_to_day,
        day_data_map=day_data_map,
        all_keys=all_fuels,
        fill_zero_for_missing=True
    )

    # ✅ Define and extract price-related dictionaries
    SpotPrice = extract_series_for_column(["Day-ahead Price (EUR/MWh)"], node_to_day, day_data_map)["Day-ahead Price (EUR/MWh)"]
    IntradayPrice = extract_series_for_column(["Intraday price (EUR/MWh)"], node_to_day, day_data_map)["Intraday price (EUR/MWh)"]
    ActivationUpPrice = extract_series_for_column(["Activation price up (mFRR)"], node_to_day, day_data_map)["Activation price up (mFRR)"]
    ActivationDwnPrice = extract_series_for_column(["Activation price down (mFRR)"], node_to_day, day_data_map)["Activation price down (mFRR)"]
    CapacityUpPrice = extract_series_for_column(["Capacity price up (mFRR)"], node_to_day, day_data_map)["Capacity price up (mFRR)"]
    CapacityDwnPrice = extract_series_for_column(["Capacity price down (mFRR)"], node_to_day, day_data_map)["Capacity price down (mFRR)"]
    PV_data = extract_series_for_column(["Soldata"], node_to_day, day_data_map)["Soldata"]
    Res_CapacityUpVolume = extract_series_for_column(["Res_Cap_Volume_Up"], node_to_day, day_data_map)["Res_Cap_Volume_Up"]
    Res_CapacityDwnVolume = extract_series_for_column(["Res_Cap_Volume_Down"], node_to_day, day_data_map)["Res_Cap_Volume_Down"]
    ID_Capacity_Sell_Volume = extract_series_for_column(["ID_Cap_Volume_Sell"], node_to_day, day_data_map)["ID_Cap_Volume_Sell"]
    ID_Capacity_Buy_Volume = extract_series_for_column(["ID_Cap_Volume_Buy"], node_to_day, day_data_map)["ID_Cap_Volume_Buy"]






    #Create Tech_availability:

    Tech_availability = {
        "PV": PV_data,
        "Power_Grid": 1.0,
        "ElectricBoiler": 0.98,
        "HP_LT": 0.98,
        "HP_MT": 0.98,
        "P2G": 0.98,
        "G2P": 0.98,
        "GasBoiler": 0.98,
        "GasBoiler_CCS": 0.98,
        "CHP": 0.8,
        "CHP_CCS": 0.8,
        "Biogas_Grid": 0.9,
        "CH4_Grid": 0.8,
        "CH4_H2_Mixer": 1.0,
        "DieselReserveGenerator": 0.98,
        "H2_Grid": 0.8,
        "Dummy_Grid": 1.0,
    }



    import pprint

    def average_dict_values(nested_dict):
        total = 0
        count = 0
        for node_data in nested_dict.values():
            for hour_value in node_data.values():
                total += hour_value
                count += 1
        return total / count if count > 0 else 0

    avg_capacity_up = average_dict_values(Res_CapacityUpVolume)
    avg_capacity_down = average_dict_values(Res_CapacityDwnVolume)

    print(f"Average Capacity Up Price: {avg_capacity_up:.2f} EUR/MW")
    print(f"Average Capacity Down Price: {avg_capacity_down:.2f} EUR/MW")

    parent_mapping = create_parent_mapping("Set_ParentCoupling.tab")

    # Preview one node per fuel
    #pprint.pprint({k: list(v.items())[:1] for k, v in ReferenceDemand.items()})
    #pggint.pprint({k: list(v.items())[:1] for k, v in Res_CapacityDwnVolume.items()})
    #pprjthgiprint({k: list(v.items())[:1] for k, v in ID_Capacity_Buy_Volume.items()})

    

    #####################################################################################
    ########################### PARAMETER GENERATION FUNCTIONS ##########################
    #####################################################################################

    # Function to count number of periods from Set_of_Periods.tab
    def get_number_of_periods_from_tab(filename="Set_of_Periods.tab", in_sample_data_folder = in_sample_data_folder):
        filepath = os.path.join(in_sample_data_folder, filename)
        df = pd.read_csv(filepath, sep="\t")
        return len(df)
    
    def generate_set_of_LoadShiftingPeriod(filename="Set_of_LoadShiftingPeriod.tab"):
        def data_generator():
            n_periods = get_number_of_periods_from_tab()
            df = pd.DataFrame({
            "LoadShiftingPeriod": list(range(1, n_periods + 1))
            })
            yield df
            """
            if excel_path == "NO1_Aluminum_2024_combined historical data.xlsx":
                # Use all periods
                df_all = pd.DataFrame({"LoadShiftingPeriod": df_periods["Periods"]})
                yield df_all
            elif excel_path == "NO1_Pulp_Paper_2024_combined historical data.xlsx" or excel_path == "NO1_Pulp_Paper_2024_combined historical data_Uten_SatSun.xlsx":
                # Use only the last period
                last_period = df_periods["Periods"].max()
                df_last = pd.DataFrame({"LoadShiftingPeriod": [last_period]})
                yield df_last
            else:
                raise ValueError(f"Unknown excel_path: {excel_path}")
            """

        # Use the make_tab_file function to write the result
        make_tab_file(filename, data_generator())

    def generate_set_of_PeriodsInMonth(branch_counts, filename="Set_of_PeriodsInMonth.tab"):
        def data_generator():
            # Extract valid period indices (starting from stage 2 = index 1)
            valid_periods = [i for i, count in enumerate(branch_counts[1:], start=1) if count > 0]

            # Create the DataFrame assigning all periods to Month 1
            df = pd.DataFrame({
                "Month": [1] * len(valid_periods),
                "PeriodInMonth": list(range(1, len(valid_periods) + 1))
            })

            yield df

        make_tab_file(filename, data_generator())

    # Functions to scale and write .tab files for each parameter
    def generate_Par_CostExpansion_Tec(filename="Par_CostExpansion_Tec.tab"):
        def data_generator():
            num_periods = get_number_of_periods_from_tab()
            rows = [
                {"Technology": tech, "CostExpansion": cost * num_periods}
                for tech, cost in CostExpansion_Tec.items()
            ]
            yield pd.DataFrame(rows)

        make_tab_file(filename, data_generator())
        

    def generate_Par_CostExpansion_Bat(filename="Par_CostExpansion_Bat.tab"):
        def data_generator():
            num_periods = get_number_of_periods_from_tab()
            rows = [
                {"StorageTech": bat, "CostExpansion": cost * num_periods}
                for bat, cost in CostExpansion_Bat.items()
            ]
            yield pd.DataFrame(rows)

        make_tab_file(filename, data_generator())


    def generate_Par_CostGridTariff(filename="Par_CostGridTariff.tab"):
        def data_generator():
            num_periods = get_number_of_periods_from_tab()
            total_tariff = CostGridTariff * num_periods

            df = pd.DataFrame([{
                "Tariff": total_tariff
            }])
            yield df
        make_tab_file(filename, data_generator())



    def generate_Par_LastPeriodInMonth(filename="Par_LastPeriodInMonth.tab"):
        def data_generator():
            # This function assumes that the number of periods is constant for all months
            num_periods = get_number_of_periods_from_tab()
            df = pd.DataFrame([{"Month": 1, "LastPeriodInMonth": num_periods}])
            yield df

        make_tab_file(filename, data_generator())
        


  

    ##########
    ## Additional Parameter generation functions ###
    ##########
    def generate_cost_activity(num_nodes, num_timesteps, cost_activity, filename="Par_ActivityCost.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
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

        make_tab_file(filename, data_generator())

    
    def generate_CapacityUpPrice(num_nodes, num_timesteps, CapacityUpPrice, filename = "Par_aFRR_UP_CAP_price.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(1, num_nodes - num_nodesInlastStage + 1):
                # For a given node, retrieve its time-dependent prices (defaults to an empty dict if not found)
                node_prices = CapacityUpPrice.get(node, {})
                for t in range(1, num_timesteps + 1):
                    price = node_prices.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "CapacityUpPrice": price})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        
        make_tab_file(filename, data_generator())

    def generate_CapacityDownPrice(num_nodes, num_timesteps, CapacityDownPrice, filename = "Par_aFRR_DWN_CAP_price.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(1, num_nodes - num_nodesInlastStage + 1):
                # For a given node, retrieve its time-dependent prices (defaults to an empty dict if not found)
                node_prices = CapacityDownPrice.get(node, {})
                for t in range(1, num_timesteps + 1):
                    price = node_prices.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "CapacityDownPrice": price})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        
        make_tab_file(filename, data_generator())


    def generate_ActivationUpPrice(num_nodes, num_timesteps, ActivationUpPrice, filename = "Par_aFRR_UP_ACT_price.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
                # Retrieve the time-dependent activation up prices for the current node (defaults to an empty dict if not found)
                node_prices = ActivationUpPrice.get(node, {})
                for t in range(1, num_timesteps + 1):
                    price = node_prices.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "ActivationUpPrice": price})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        
        make_tab_file(filename, data_generator())

    def generate_ActivationDownPrice(num_nodes, num_timesteps, ActivationDownPrice, filename = "Par_aFRR_DWN_ACT_price.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
                # Retrieve the time-dependent activation down prices for the current node (defaults to an empty dict if not found)
                node_prices = ActivationDownPrice.get(node, {})
                for t in range(1, num_timesteps + 1):
                    price = node_prices.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "ActivationDownPrice": price})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        
        make_tab_file(filename, data_generator())

    def generate_SpotPrice(num_nodes, num_timesteps, SpotPrice, filename = "Par_SpotPrice.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
                # Retrieve time-dependent spot prices for the current node (defaults to an empty dict if not found)
                node_prices = SpotPrice.get(node, {})
                for t in range(1, num_timesteps + 1):
                    price = node_prices.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "Spot_Price": price})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        
        make_tab_file(filename, data_generator())

    def generate_IntradayPrice(num_nodes, num_timesteps, IntradayPrice, filename = "Par_IntradayPrice.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
                # Retrieve time-dependent spot prices for the current node (defaults to an empty dict if not found)
                node_prices = IntradayPrice.get(node, {})
                for t in range(1, num_timesteps + 1):
                    price = node_prices.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "Intraday_Price": price})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        
        make_tab_file(filename, data_generator())


    def generate_ReferenceDemand(num_nodes, num_timesteps, energy_carriers, ReferenceDemand, filename = "Par_EnergyDemand.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            # Loop over nodes (using the same range as cost_export for consistency)
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
                for ec in energy_carriers:
                    for t in range(1, num_timesteps + 1):
                        ec_value = ReferenceDemand.get(ec, 0.0)
                        # If ec_value is a dict and contains a node-specific entry, use that
                        if isinstance(ec_value, dict) and (node in ec_value):
                            node_value = ec_value[node]
                            if isinstance(node_value, dict):
                                demand = node_value.get(t, 0.0)
                            elif isinstance(node_value, list):
                                demand = node_value[t - 1] if len(node_value) >= t else 0.0
                            else:
                                demand = node_value
                        else:
                            # Otherwise treat ec_value as time-dependent or constant
                            if isinstance(ec_value, dict):
                                demand = ec_value.get(t, 0.0)
                            elif isinstance(ec_value, list):
                                demand = ec_value[t - 1] if len(ec_value) >= t else 0.0
                            else:
                                demand = ec_value
                        rows.append({"Node": node, "Time": t, "EnergyCarrier": ec, "ReferenceDemand": demand})
                        count += 1
                        if count % chunk_size == 0:
                            yield pd.DataFrame(rows)
                            rows = []
            if rows:
                yield pd.DataFrame(rows)
        make_tab_file(filename, data_generator())

    def generate_NodeProbability(num_nodes, NodeProbability, filename = "Par_NodesProbability.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            # Loop over each node (assumed to be numbered 1 to num_nodes)
            for node in range(1, num_nodes + 1):
                prob = NodeProbability.get(node, 0.0)
                rows.append({"Node": node, "NodeProbability": prob})
                count += 1
                if count % chunk_size == 0:
                    yield pd.DataFrame(rows)
                    rows = []
            if rows:
                yield pd.DataFrame(rows)
        make_tab_file(filename, data_generator())

    def generate_availability_factor(num_nodes, num_timesteps, technologies, tech_availability, filename="Par_AvailabilityFactor.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            # Note: The following loop starts at num_firstStageNodes + 1.
            # If not needed, you can replace the range with: range(1, num_nodes + 1)
            for node in range(num_firstStageNodes + 1, num_nodes + 1):
                for tech in technologies:
                    for t in range(1, num_timesteps + 1):
                        if tech == 'PV' and isinstance(tech_availability.get(tech), dict):
                            # For PV, retrieve the node- and time-specific factor.
                            avail = tech_availability[tech].get(node, {}).get(t, 0.0)
                        else:
                            # For other technologies, use the constant value.
                            avail = tech_availability.get(tech, 0.0)
                        rows.append({
                            "Node": node,
                            "Time": t,
                            "Technology": tech,
                            "AvailabilityFactor": avail
                        })
                        count += 1
                        if count % chunk_size == 0:
                            yield pd.DataFrame(rows)
                            rows = []
            if rows:
                yield pd.DataFrame(rows)

        make_tab_file(filename, data_generator())



    def generate_activation_factors(num_nodes, num_timesteps, parent_mapping, activation_rate=0.8):
        
        #activation_rate = andel av barnenodene som skal være aktive (enten opp eller ned) i hver time.
        
        parent_to_children = {}
        for child, parent in parent_mapping.items():
            parent_to_children.setdefault(parent, []).append(child)

        rows = []

        for parent, children in parent_to_children.items():
            for t in range(1, num_timesteps + 1):
                num_children = len(children)
                if num_children < 2:
                    raise ValueError(f"Forelder {parent} har for få barn til å sikre både opp- og nedregulering i time {t}")

                # Hvor mange barn skal aktiveres denne timen?
                num_active = max(2, int(activation_rate * num_children))  # minst 2 (én opp, én ned)
                
                active_children = random.sample(children, num_active)

                # Plukk én for opp, én for ned
                random.shuffle(active_children)
                child_up = active_children.pop()
                child_down = active_children.pop()

                activation = {}

                for child in children:
                    if child == child_up:
                        activation[child] = (1, 0)
                    elif child == child_down:
                        activation[child] = (0, 1)
                    elif child in active_children:
                        # Fordel tilfeldig opp eller ned på resten av de aktive
                        if random.random() < 0.5:
                            activation[child] = (1, 0)
                        else:
                            activation[child] = (0, 1)
                    else:
                        # Ikke aktivert
                        activation[child] = (0, 0)

                for child, (up, down) in activation.items():
                    rows.append({
                        "Node": child,
                        "Time": t,
                        "ActivationFactorUpReg": up,
                        "ActivationFactorDownReg": down
                    })

        return pd.DataFrame(rows)

    ################################################################################################
    ########################### ACTIVATION FACTORS FOR ALU-INDUSTRY ################################
    ################################################################################################
    # Krever ca. 50 branches for 30% aktiveringsrate med 8t hviletid - Færre branches krever activation_rate.
    # For få branches vil gi en feilmelding (For få tilgjengelige barn...), så bare å prøve seg frem:)

    def generate_activation_factors_with_rest_time(num_nodes, num_timesteps, parent_mapping, activation_rate=0.10, rest_hours=8):
    
        #Generate activation factors with 8 hours rest after activation.
        
        parent_to_children = {}
        for child, parent in parent_mapping.items():
            parent_to_children.setdefault(parent, []).append(child)

        rows = []
        
        # Track when each child node becomes available again
        available_from = {child: 1 for child in parent_mapping.keys()}  # Starter som tilgjengelig fra time 1

        for parent, children in parent_to_children.items():
            for t in range(1, num_timesteps + 1):
                # Filter children that are available at time t
                available_children = [child for child in children if available_from[child] <= t]

                if len(available_children) < 2:
                    raise ValueError(f"For få tilgjengelige barn for forelder {parent} ved time {t} for å sikre aktivering.")

                # Hvor mange skal aktiveres
                num_children = len(available_children)
                num_active = max(2, int(activation_rate * num_children))  # minst 2

                active_children = random.sample(available_children, min(num_active, num_children))

                random.shuffle(active_children)
                child_up = active_children.pop()
                child_down = active_children.pop()

                activation = {}

                for child in children:
                    if child == child_up:
                        activation[child] = (1, 0)
                        available_from[child] = t + rest_hours + 1  # Låst i 8 timer etter aktivering
                    elif child == child_down:
                        activation[child] = (0, 1)
                        available_from[child] = t + rest_hours + 1
                    elif child in active_children:
                        if random.random() < 0.5:
                            activation[child] = (1, 0)
                        else:
                            activation[child] = (0, 1)
                        available_from[child] = t + rest_hours + 1
                    else:
                        activation[child] = (0, 0)

                # Logg aktiveringer
                for child, (up, down) in activation.items():
                    rows.append({
                        "Node": child,
                        "Time": t,
                        "ActivationFactorUpReg": up,
                        "ActivationFactorDownReg": down
                    })

        return pd.DataFrame(rows)
    
    def generate_max_upshift_file(excel_path, num_timesteps, filename="Par_MaxUpShift.tab"):
        if "Pulp_Paper" in excel_path:
            factor = 0.1
            industry = "pulp"
        elif "Aluminum" in excel_path:
            factor = 0.05
            industry = "alu"
        else:
            raise ValueError("Invalid excel_path: must contain 'Pulp_Paper' or 'Aluminum'")

        def data_generator():
            shift_hours = range(8, 18)
            rows = []
            for t in range(1, num_timesteps + 1):
                if industry == "alu":
                    value = factor
                elif industry == "pulp":
                    value = factor if t in shift_hours else 0
                rows.append({"Time": t, "MaximumUpShift": value})
            yield pd.DataFrame(rows)

        make_tab_file(filename, data_generator())
        

    def generate_max_downshift_file(excel_path, num_timesteps, filename="Par_MaxDwnShift.tab"):
        shift_hours = range(8, 18)

        if "Pulp_Paper" in excel_path:
            factor = 0.3
            industry = "pulp"
        elif "Aluminum" in excel_path:
            factor = 0.2
            industry = "alu"
        else:
            raise ValueError("Invalid excel_path: must contain 'Pulp_Paper' or 'Aluminum'")

        def data_generator():
            shift_hours = range(8, 18)
            rows = []
            for t in range(1, num_timesteps + 1):
                if industry == "alu":
                    value = factor
                elif industry == "pulp":
                    value = factor if t in shift_hours else 0
                rows.append({"Time": t, "MaximumUpShift": value})
            yield pd.DataFrame(rows)

        make_tab_file(filename, data_generator())
        

    


    def generate_joint_regulation_activation_files(num_nodes, num_timesteps, up_filename = "Par_ActivationFactor_Up_Reg.tab", down_filename = "Par_ActivationFactor_Dwn_Reg.tab"):
        if excel_path == "NO1_Pulp_Paper_2024_combined historical data.xlsx" or excel_path == "NO1_Pulp_Paper_2024_combined historical data_Uten_SatSun.xlsx":
            df_joint = generate_activation_factors(num_nodes, num_timesteps, parent_mapping)
        elif excel_path == "NO1_Aluminum_2024_combined historical data.xlsx":
            df_joint = generate_activation_factors(num_nodes, num_timesteps, parent_mapping)
        else:
            raise ValueError("Invalid excel_path. Please provide a valid path.")

        # Write UpReg file
        def data_generator_up():
            df_up = df_joint.rename(columns={"ActivationFactorUpReg": "ActivationFactorUpRegulation"})
            yield df_up[["Node", "Time", "ActivationFactorUpRegulation"]]

        make_tab_file(up_filename, data_generator_up())

        # Write DownReg file
        def data_generator_down():
            df_down = df_joint.rename(columns={"ActivationFactorDownReg": "ActivationFactorDwnRegulation"})
            yield df_down[["Node", "Time", "ActivationFactorDwnRegulation"]]

        make_tab_file(down_filename, data_generator_down())



    def generate_ID_factors(num_nodes, num_timesteps, p_id_up=0.4, p_id_down=0.4):
        rows = []
        # Process nodes beyond the first-stage nodes.
        for node in range(num_firstStageNodes + 1, num_nodes + 1):
            for t in range(1, num_timesteps + 1):
                # Independent draws for ID up and ID down.
                up = 1 if random.random() < p_id_up else 0
                down = 1 if random.random() < p_id_down else 0
                
                rows.append({
                    "Node": node,
                    "Time": t,
                    "ActivationFactorID_UP": up,
                    "ActivationFactorID_Dwn": down
                })
        return pd.DataFrame(rows)

    def generate_ActivationFactorID_UP(num_nodes, num_timesteps, p_id_up = 0.3, p_id_down = 0.2, filename = "Par_ActivationFactor_ID_Up_Reg.tab"):
        def data_generator(chunk_size=10_000_000):
            df_joint = generate_ID_factors(num_nodes, num_timesteps, p_id_up, p_id_down)
            # Rename column for clarity
            df_joint = df_joint.rename(columns={"ActivationFactorID_UP": "ActivationFactorID_UP_Reg"})
            yield df_joint[["Node", "Time", "ActivationFactorID_UP_Reg"]]
        
        make_tab_file(filename, data_generator())

    def generate_ActivationFactorID_Dwn(num_nodes, num_timesteps, p_id_up = 0.3, p_id_down = 0.2, filename = "Par_ActivationFactor_ID_Dwn_Reg.tab"):
        def data_generator(chunk_size=10_000_000):
            df_joint = generate_ID_factors(num_nodes, num_timesteps, p_id_up, p_id_down)
            df_joint = df_joint.rename(columns={"ActivationFactorID_Dwn": "ActivationFactorID_Dwn_Reg"})
            yield df_joint[["Node", "Time", "ActivationFactorID_Dwn_Reg"]]
        
        make_tab_file(filename, data_generator())

    def generate_Res_CapacityUpVolume(num_nodes, num_timesteps, Res_CapacityUpVolume, filename="Par_Res_CapacityUpVolume.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(1, num_nodes + 1):
                node_data = Res_CapacityUpVolume.get(node, {})
                for t in range(1, num_timesteps + 1):
                    volume = node_data.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "Res_CapacityUpVolume": volume})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        make_tab_file(filename, data_generator())


    def generate_Res_CapacityDownVolume(num_nodes, num_timesteps, Res_CapacityDwnVolume, filename="Par_Res_CapacityDownVolume.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(1, num_nodes + 1):
                node_data = Res_CapacityDwnVolume.get(node, {})
                for t in range(1, num_timesteps + 1):
                    volume = node_data.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "Res_CapacityDownVolume": volume})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        make_tab_file(filename, data_generator())

    def generate_ID_Capacity_Sell_Volume(num_nodes, num_timesteps, Res_CapacityDwnVolume, filename="Par_ID_Capacity_Sell_Volume.tab"):
        def data_generator(chunk_size=10_000_000):
            rows = []
            count = 0
            for node in range(1, num_nodes + 1):
                node_data = ID_Capacity_Sell_Volume.get(node, {})
                for t in range(1, num_timesteps + 1):
                    volume = node_data.get(t, 0.0)
                    rows.append({"Node": node, "Time": t, "ID_Capacity_Sell_Volume": volume})
                    count += 1
                    if count % chunk_size == 0:
                        yield pd.DataFrame(rows)
                        rows = []
            if rows:
                yield pd.DataFrame(rows)
        make_tab_file(filename, data_generator())

    def generate_ID_Capacity_Buy_Volume(num_nodes, num_timesteps, Res_CapacityDwnVolume, filename="Par_ID_Capacity_Buy_Volume.tab"):
            def data_generator(chunk_size=10_000_000):
                rows = []
                count = 0
                for node in range(1, num_nodes + 1):
                    node_data = ID_Capacity_Buy_Volume.get(node, {})
                    for t in range(1, num_timesteps + 1):
                        volume = node_data.get(t, 0.0)
                        rows.append({"Node": node, "Time": t, "ID_Capacity_Buy_Volume": volume})
                        count += 1
                        if count % chunk_size == 0:
                            yield pd.DataFrame(rows)
                            rows = []
                if rows:
                    yield pd.DataFrame(rows)
            make_tab_file(filename, data_generator())



    ##########################################################################
    ########################### GENERATE SETS ################################
    ##########################################################################
    generate_Set_TimeSteps(num_timesteps)
    generate_Set_of_Nodes(num_nodes)
    generate_set_of_Parents(num_nodes)
    generate_set_of_NodesInStage([num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage, num_branches_to_eleventhStage, num_branches_to_twelfthStage, num_branches_to_thirteenthStage, num_branches_to_fourteenthStage, num_branches_to_fifteenthStage])
    generate_set_of_Periods([num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage, num_branches_to_eleventhStage, num_branches_to_twelfthStage, num_branches_to_thirteenthStage, num_branches_to_fourteenthStage, num_branches_to_fifteenthStage])
    generate_set_of_PeriodsInMonth([num_branches_to_firstStage, num_branches_to_secondStage, num_branches_to_thirdStage, num_branches_to_fourthStage, num_branches_to_fifthStage, num_branches_to_sixthStage, num_branches_to_seventhStage, num_branches_to_eighthStage, num_branches_to_ninthStage, num_branches_to_tenthStage, num_branches_to_eleventhStage, num_branches_to_twelfthStage, num_branches_to_thirteenthStage, num_branches_to_fourteenthStage, num_branches_to_fifteenthStage])
    generate_set_of_LoadShiftingPeriod()
    generate_set_of_NodesFirst(num_branches_to_firstStage)

    ##########################################################################
    ########################### GENERATE PARAMETERS ##########################
    ##########################################################################

    generate_cost_activity(num_nodes, num_timesteps, cost_activity)
    generate_CapacityUpPrice(num_nodes, num_timesteps, CapacityUpPrice)
    generate_CapacityDownPrice(num_nodes, num_timesteps, CapacityDwnPrice)
    generate_ActivationUpPrice(num_nodes, num_timesteps, ActivationUpPrice)
    generate_ActivationDownPrice(num_nodes, num_timesteps, ActivationDwnPrice)
    generate_SpotPrice(num_nodes, num_timesteps, SpotPrice)
    generate_IntradayPrice(num_nodes, num_timesteps, IntradayPrice)
    generate_ReferenceDemand(num_nodes, num_timesteps, energy_carriers, ReferenceDemand)
    generate_NodeProbability(num_nodes, NodeProbability)
    generate_availability_factor(num_nodes, num_timesteps, technologies, Tech_availability)
    generate_joint_regulation_activation_files(num_nodes, num_timesteps)
    generate_ActivationFactorID_UP(num_nodes, num_timesteps, p_id_up=0.3, p_id_down=0.2)
    generate_ActivationFactorID_Dwn(num_nodes, num_timesteps, p_id_up=0.3, p_id_down=0.2)
    generate_Res_CapacityDownVolume(num_nodes, num_timesteps, Res_CapacityDwnVolume)
    generate_Res_CapacityUpVolume(num_nodes, num_timesteps, Res_CapacityUpVolume)
    generate_ID_Capacity_Sell_Volume(num_nodes, num_timesteps, ID_Capacity_Sell_Volume)
    generate_ID_Capacity_Buy_Volume(num_nodes, num_timesteps, ID_Capacity_Buy_Volume)
    generate_max_upshift_file(excel_path, num_timesteps=24)
    generate_max_downshift_file(excel_path, num_timesteps=24)


  # Call them after Set_of_Periods.tab is created
    generate_Par_CostExpansion_Tec()
    generate_Par_CostExpansion_Bat()
    generate_Par_CostGridTariff()
    generate_Par_LastPeriodInMonth()