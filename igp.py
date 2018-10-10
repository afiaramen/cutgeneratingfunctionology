## Module

from sage.all import *
del SetPartitionsAk
del SetPartitionsBk
del SetPartitionsIk
del SetPartitionsPRk
del SetPartitionsPk
del SetPartitionsRk
del SetPartitionsSk
del SetPartitionsTk


igp_dir = os.path.dirname(__file__)
if igp_dir:
    igp_dir += "/"

load(igp_dir + "logging.sage")
load(igp_dir + "intervals.sage")
load(igp_dir + "real_number_field.sage")
load(igp_dir + "fast_linear.sage")
load(igp_dir + "functions.sage")
load(igp_dir + "continuous_case.sage")
load(igp_dir + "discontinuous_case.sage")
load(igp_dir + "discrete_case.sage")
load(igp_dir + "compendium_procedures.sage")
load(igp_dir + "extreme_functions_in_literature.sage")
load(igp_dir + "extreme_functions_sporadic.sage")
load(igp_dir + "survey_examples.sage")
load(igp_dir + "simple_extremality_test.sage")
load(igp_dir + "semialgebraic_mathematica.sage")
load(igp_dir + "parametric.sage")
load(igp_dir + "parametric_cpl.sage")
load(igp_dir + "kslope_ppl_mip.py")
load(igp_dir + "vertex_enumeration.py")
load(igp_dir + "quasi_periodic.sage")
load(igp_dir + "extreme_functions_mlr_cpl3.sage")
load(igp_dir + "kslope_pattern.sage")
load(igp_dir + "2q_mip.sage")
load(igp_dir + "kslope_mip.sage")
#load("old_walk_and_stability_interval_code.sage")
load(igp_dir + "animation_2d_diagram.sage")
load(igp_dir + "crazy_perturbation.sage")
load(igp_dir + "crazy_perturbation_examples.sage")
load(igp_dir + "bug_examples.sage")
load(igp_dir + "lifting_project.sage")
load(igp_dir + "plot_options.sage")
# Dual feasible functions
load(igp_dir + "gdff_linear_test.sage")
load(igp_dir + "dff_functions.sage")
load(igp_dir + "dff_test_plot.sage")
load(igp_dir + "discontinuous_dff.sage")
load(igp_dir + "computer_based_search_naive_dff.sage")
#load(igp_dir + "Gomory_conversion.sage")

# multirow
load(igp_dir + "piecewise_functions.sage")
load(igp_dir + "lifting_region.sage")

# faster subaddtivity function
load(igp_dir + "faster_subadditivity_test.sage")
load(igp_dir + "test_functions.sage")

import extreme_functions, procedures

try:
    load(igp_dir + "config.sage")
except IOError:
    pass

logging.info("Welcome to the infinite-group-relaxation-code. DON'T PANIC. See demo.sage for instructions.")
