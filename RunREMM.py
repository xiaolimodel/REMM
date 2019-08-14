from urbansim_defaults import datasources, models, utils
import datasources, models, variables, utils
import urbansim.sim.simulation as sim
import pandas as pd
import numpy as np
import os

#sim.run(["travel_time_reset"])
sim.run([
    "clear_cache",            # recompute variables every year
    "run_arcpy",
    "neighborhood_vars",      # neighborhood variables
    "households_transition",  # households transition
    #"households_relocation",  # households relocation model
    "jobs_transition",        # jobs transition
    #"jobs_relocation",        # jobs relocation model
    "nrh_ind_simulate",       # industrial price model
    "nrh_ofc_simulate",       # office price model
    "nrh_ret_simulate",       # retail price model
    "nrh_mu_oth_simulate",    # non-residential mixed-use, other model
    "rsh_sf_simulate",        # single-family residential price model
    "rsh_mf_simulate",        # multi-family residential price model
    "hlcm_simulate_slc",      # households location choice Salt Lake County
    "hlcm_simulate_utah",     # households location choice Utah County
    "hlcm_simulate_davis",    # households location choice Davis County
    "hlcm_simulate_weber",    # households location choice Weber County
    "elcm_simulate_slc",      # employment location choice Salt Lake County
    "elcm_simulate_utah",     # employment location choice Utah County
    "elcm_simulate_davis",    # employment location choice Davis County
    "elcm_simulate_weber",    # employment location choice Weber County
    #"clear_cache",
    "indicator_export",       # export county and zone level indicators to csv
    "travel_model_export_no_construction",
    #"trend_calibration",
    #"network_prices",
    "feasibility",            # compute development feasibility
    "garbage_collect",
    "residential_developer_slc" ,  # build actual buildings Salt Lake County
    "residential_developer_utah",  # build actual buildings Utah County
    "residential_developer_davis",  # build actual buildings Davis County
    "residential_developer_weber",  # build actual buildings Weber County
    "office_developer_slc",
    "office_developer_utah",
    "office_developer_davis",
    "office_developer_weber",
    "retail_developer_slc",
    "retail_developer_utah",
    "retail_developer_davis",
    "retail_developer_weber",
    "industrial_developer_slc",
    "industrial_developer_utah",
    "industrial_developer_davis",
    "industrial_developer_weber",
    "travel_model_export_add_construction",           # export travel model inputs at TAZ level in specified years
    #"travel_time_import",
    "run_cube",               # call Cube and import travel times in specified years
], years=range(2015,2061))  #, data_out=utils.get_run_filename(), out_interval=10)

