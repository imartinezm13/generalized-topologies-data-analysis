import os
import pandas as pd
from utils.topo import closure, interior

def write_to_excel(results_dict, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for key, value in results_dict.items():
            value.to_excel(writer, sheet_name=key, index=False)

def refine_base(subset, new_base, reference_base):
    refined_base = [group for group in new_base if set(group) & set(subset)]
    covered_elements = set(elem for group in refined_base for elem in group)
    for group in reference_base:
        if not all(elem in covered_elements for elem in group):
            refined_base.append(group)
            covered_elements.update(group)

    return refined_base

def process_closures_against_base(
    dataframe, 
    base_group, 
    target_groups: dict,
    A_param,
    output_file="excel_results/result.xlsx"
):
    base_int = [int(i) for i in base_group]
    closure_results = {}
    extras = {}

    gen_base_1_re = refine_base(base_group, target_groups["gen_base_1"], target_groups["base"])
    gen_base_2_re = refine_base(base_group, target_groups["gen_base_2"], target_groups["base"])

    target_groups_refined = {
        "gen_base_1": gen_base_1_re,
        "gen_base_2": gen_base_2_re,
        "base": target_groups["base"],
        "clusters": target_groups["clusters"]
    }

    for name, group in target_groups_refined.items():
        highlight = [int(i) for i in closure(group, base_group, A_param)]
        closure_results[name] = highlight
        extras[name] = list(set(highlight) - set(base_int))

        print(f"{name} extra:", dataframe["Country"].iloc[extras[name]].tolist())

    # Prepare Excel output
    results = {}
    for name in closure_results:
        results[f"all_{name}"] = dataframe.iloc[closure_results[name]]
        results[f"extra_closure_{name}"] = dataframe.iloc[extras[name]]

    write_to_excel(results, output_file)


def process_interiors_against_base(
    dataframe, 
    base_group, 
    target_groups: dict,
    output_file="excel_results/result.xlsx"
):
    base_int = [int(i) for i in base_group]
    interior_results = {}

    gen_base_1_re = refine_base(base_group, target_groups["gen_base_1"], target_groups["base"])
    gen_base_2_re = refine_base(base_group, target_groups["gen_base_2"], target_groups["base"])

    target_groups_refined = {
        "gen_base_1": gen_base_1_re,
        "gen_base_2": gen_base_2_re,
        "base": target_groups["base"],
        "clusters": target_groups["clusters"]
    }

    for name, group in target_groups_refined.items():
        highlight = [int(i) for i in interior(group, base_group)]
        interior_results[name] = highlight

        # Optional: print representative members
        print(f"{name} representative (interior):", dataframe["Country"].iloc[highlight].tolist())

    # Prepare Excel output
    results = {}
    for name in interior_results:
        results[f"base_{name}"] = dataframe.iloc[base_int]
        results[f"interior_{name}"] = dataframe.iloc[interior_results[name]]

    write_to_excel(results, output_file)    

# General function to process a batch
def process_batch(group_dict, datos, target_groups, A, output_folder="results"):
    os.makedirs(output_folder, exist_ok=True)
    for base_name, base_group in group_dict.items():
        print(f"Processing {base_name}...")

        process_closures_against_base(
            dataframe=datos,
            base_group=base_group,
            target_groups=target_groups,
            A_param=A,
            output_file=f"{output_folder}/closure_{base_name}.xlsx"
        )

        print("")

        process_interiors_against_base(
            dataframe=datos,
            base_group=base_group,
            target_groups=target_groups,
            output_file=f"{output_folder}/interior_{base_name}.xlsx"
        )
        print("")
