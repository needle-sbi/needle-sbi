#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import io
import logging
import os

import numpy as np
import pandas as pd
from numpy import cos, cosh, exp, sin, sinh, sqrt

# Get the logging level from an environment variable, default to INFO
log_level = os.getenv("LOG_LEVEL", "INFO").upper()


logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),  # Fallback to INFO if the level is invalid
    format="%(asctime)s - %(name)-20s - %(levelname) -8s - %(message)s",
)

logger = logging.getLogger(__name__)

__doc__ = """
This module contains the functions to calculate the derived quantities of the HEP dataset.
Originally written by David Rousseau, and Victor Estrade.
"""
__version__ = "4.0"
__author__ = "David Rousseau, and Victor Estrade "


def calcul_int(data):
    """
    Calculate the px py pz E components of the particles' 4 momentum.

    Args:
        data (pandas.DataFrame): Input data containing the particle properties.

    Returns:
        pandas.DataFrame: Dataframe with the derived quantities calculated.
    """
    # Definition of the x and y components of the hadron's momentum
    data["had_px"] = data.PRI_had_pt * cos(data.PRI_had_phi)
    data["had_py"] = data.PRI_had_pt * sin(data.PRI_had_phi)
    data["had_pz"] = data.PRI_had_pt * sinh(data.PRI_had_eta)
    data["p_had"] = data.PRI_had_pt * cosh(data.PRI_had_eta)

    # Definition of the x and y components of the lepton's momentum
    data["lep_px"] = data.PRI_lep_pt * cos(data.PRI_lep_phi)
    data["lep_py"] = data.PRI_lep_pt * sin(data.PRI_lep_phi)
    data["lep_pz"] = data.PRI_lep_pt * sinh(data.PRI_lep_eta)
    data["p_lep"] = data.PRI_lep_pt * cosh(data.PRI_lep_eta)

    # Definition of the x and y components of the neutrinos's momentum (MET)
    data["met_x"] = data.PRI_met * cos(data.PRI_met_phi)
    data["met_y"] = data.PRI_met * sin(data.PRI_met_phi)

    # Definition of the x and y components of the leading jet's momentum
    data["jet_leading_px"] = (
        data.PRI_jet_leading_pt * cos(data.PRI_jet_leading_phi) * (data.PRI_n_jets >= 1)
    )  # = 0 if PRI_n_jets == 0
    data["jet_leading_py"] = data.PRI_jet_leading_pt * sin(data.PRI_jet_leading_phi) * (data.PRI_n_jets >= 1)
    data["jet_leading_pz"] = data.PRI_jet_leading_pt * sinh(data.PRI_jet_leading_eta) * (data.PRI_n_jets >= 1)
    data["p_jet_leading"] = data.PRI_jet_leading_pt * cosh(data.PRI_jet_leading_eta) * (data.PRI_n_jets >= 1)

    # Definition of the x and y components of the subleading jet's momentum
    data["jet_subleading_px"] = (
        data.PRI_jet_subleading_pt * cos(data.PRI_jet_subleading_phi) * (data.PRI_n_jets >= 2)
    )  # = 0 if PRI_n_jets <= 1
    data["jet_subleading_py"] = data.PRI_jet_subleading_pt * sin(data.PRI_jet_subleading_phi) * (data.PRI_n_jets >= 2)
    data["jet_subleading_pz"] = data.PRI_jet_subleading_pt * sinh(data.PRI_jet_subleading_eta) * (data.PRI_n_jets >= 2)
    data["p_jet_subleading"] = data.PRI_jet_subleading_pt * cosh(data.PRI_jet_subleading_eta) * (data.PRI_n_jets >= 2)

    return data


def f_DER_mass_transverse_met_lep(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the transverse mass between MET and the lepton.

    Args:
        data (pandas.DataFrame): Input dataframe with derived momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_mass_transverse_met_lep`.
    """
    data["calcul_int"] = (
        (data.PRI_met + data.PRI_lep_pt) ** 2 - (data.met_x + data.lep_px) ** 2 - (data.met_y + data.lep_py) ** 2
    )
    data["DER_mass_transverse_met_lep"] = sqrt(data.calcul_int * (data.calcul_int >= 0))
    del data["calcul_int"]
    return data


def f_DER_mass_vis(data):
    """Calculate the invariant mass of the hadron and lepton system.

    Args:
        data (pandas.DataFrame): Input dataframe with momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_mass_vis`.
    """

    data["DER_mass_vis"] = sqrt(
        (data.p_lep + data.p_had) ** 2
        - (data.lep_px + data.had_px) ** 2
        - (data.lep_py + data.had_py) ** 2
        - (data.lep_pz + data.had_pz) ** 2
    )
    return data


def f_DER_pt_h(data):
    """Calculate the transverse momentum of the hadronic system.

    Args:
        data (pandas.DataFrame): Input dataframe with momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_pt_h`.
    """

    data["DER_pt_h"] = sqrt(
        (data.had_px + data.lep_px + data.met_x) ** 2 + (data.had_py + data.lep_py + data.met_y) ** 2
    )
    return data


def f_DER_deltaeta_jet_jet(data):
    """Calculate the pseudorapidity difference between the two jets.

    Args:
        data (pandas.DataFrame): Input dataframe with jet columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_deltaeta_jet_jet`.
    """

    data["DER_deltaeta_jet_jet"] = abs(data.PRI_jet_subleading_eta - data.PRI_jet_leading_eta) * (
        data.PRI_n_jets >= 2
    ) - 25 * (data.PRI_n_jets < 2)
    return data


from numpy import sqrt


# undefined if PRI_n_jets <= 1:
def f_DER_mass_jet_jet(data):
    """Calculate the invariant mass of the two jets.

    Args:
        data (pandas.DataFrame): Input dataframe with jet momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_mass_jet_jet`.
    """

    data["calcul_int"] = (
        (data.p_jet_leading + data.p_jet_subleading) ** 2
        - (data.jet_leading_px + data.jet_subleading_px) ** 2
        - (data.jet_leading_py + data.jet_subleading_py) ** 2
        - (data.jet_leading_pz + data.jet_subleading_pz) ** 2
    )
    data["DER_mass_jet_jet"] = sqrt(data.calcul_int * (data.calcul_int >= 0)) * (data.PRI_n_jets >= 2) - 25 * (
        data.PRI_n_jets <= 1
    )

    del data["calcul_int"]
    return data


def f_DER_prodeta_jet_jet(data):
    """Calculate the product of the pseudorapidities of the two jets.

    Args:
        data (pandas.DataFrame): Input dataframe with jet columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_prodeta_jet_jet`.
    """

    data["DER_prodeta_jet_jet"] = data.PRI_jet_leading_eta * data.PRI_jet_subleading_eta * (
        data.PRI_n_jets >= 2
    ) - 25 * (data.PRI_n_jets <= 1)
    return data


def f_DER_deltar_had_lep(data):
    """Calculate the delta-R between the hadron and the lepton.

    Args:
        data (pandas.DataFrame): Input dataframe with derived momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_deltar_had_lep`.
    """
    data["difference2_eta"] = (data.PRI_lep_eta - data.PRI_had_eta) ** 2
    data["difference2_phi"] = (np.abs(np.mod(data.PRI_lep_phi - data.PRI_had_phi + 3 * np.pi, 2 * np.pi) - np.pi)) ** 2
    data["DER_deltar_had_lep"] = sqrt(data.difference2_eta + data.difference2_phi)

    del data["difference2_eta"]
    del data["difference2_phi"]
    return data


def f_DER_pt_tot(data):
    """Calculate the total transverse momentum of the event.

    Args:
        data (pandas.DataFrame): Input dataframe with momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_pt_tot`.
    """
    data["DER_pt_tot"] = sqrt(
        (data.had_px + data.lep_px + data.met_x + data.jet_leading_px + data.jet_subleading_px) ** 2
        + (data.had_py + data.lep_py + data.met_y + data.jet_leading_py + data.jet_subleading_py) ** 2
    )
    return data


def f_DER_sum_pt(data):
    """Calculate the sum of transverse momentum of the lepton, hadron, and jets.

    Args:
        data (pandas.DataFrame): Input dataframe with momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_sum_pt`.
    """

    data["DER_sum_pt"] = data.PRI_had_pt + data.PRI_lep_pt + data.PRI_jet_all_pt
    return data


def f_DER_pt_ratio_lep_had(data):
    """Calculate the transverse momentum ratio of lepton to hadron.

    Args:
        data (pandas.DataFrame): Input dataframe with momentum columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_pt_ratio_lep_had`.
    """
    data["DER_pt_ratio_lep_had"] = data.PRI_lep_pt / data.PRI_had_pt
    return data


def f_DER_met_phi_centrality(data):
    """Calculate the phi centrality of MET relative to the lepton and hadron.

    Args:
        data (pandas.DataFrame): Input dataframe with angular columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_met_phi_centrality`.
    """

    def A(met, lep, had):
        return sin(met - lep) * np.sign(sin(had - lep))

    def B(met, lep, had):
        return sin(had - met) * np.sign(sin(had - lep))

    data["A"] = A(data.PRI_met_phi, data.PRI_lep_phi, data.PRI_had_phi)
    data["B"] = B(data.PRI_met_phi, data.PRI_lep_phi, data.PRI_had_phi)
    data["num"] = data.A + data.B
    data["denum"] = sqrt(data.A**2 + data.B**2)

    data["DER_met_phi_centrality"] = data.num / (data.denum + (data.denum == 0)) * (data.denum != 0) - 25 * (
        data.denum == 0
    )
    epsilon = 0.0001
    mask = data.denum == 0

    data.loc[mask, "A"] = A(data.PRI_met_phi, data.PRI_lep_phi + epsilon, data.PRI_had_phi)
    data.loc[mask, "B"] = B(data.PRI_met_phi, data.PRI_lep_phi + epsilon, data.PRI_had_phi)
    data.loc[mask, "num"] = data.A + data.B
    data.loc[mask, "denum"] = sqrt(data.A**2 + data.B**2)
    data.loc[mask, "DER_met_phi_centrality"] = data.num / (data.denum + (data.denum == 0)) * (data.denum != 0) - 25 * (
        data.denum == 0
    )

    del data["A"]
    del data["B"]
    del data["num"]
    del data["denum"]
    return data


def f_DER_lep_eta_centrality(data):
    """Calculate the lepton eta centrality.

    Args:
        data (pandas.DataFrame): Input dataframe with jet eta columns.

    Returns:
        pandas.DataFrame: Updated dataframe with `DER_lep_eta_centrality`.
    """

    data["difference"] = (data.PRI_jet_leading_eta - data.PRI_jet_subleading_eta) ** 2
    data["moyenne"] = (data.PRI_jet_leading_eta + data.PRI_jet_subleading_eta) / 2

    epsilon = 0.0001
    mask = data["difference"] == 0.0

    data["DER_lep_eta_centrality"] = exp(-4 / (data.difference) * ((data.PRI_lep_eta - data.moyenne) ** 2)) * (
        data.PRI_n_jets >= 2
    ) - 25 * (data.PRI_n_jets <= 1)

    data.loc[mask, "DER_lep_eta_centrality"] = exp(
        -4 / (data.difference + epsilon) * ((data.PRI_lep_eta - data.moyenne) ** 2)
    ) * (data.PRI_n_jets >= 2) - 25 * (data.PRI_n_jets <= 1)

    del data["difference"]
    del data["moyenne"]

    return data


def f_del_DER(data):
    """Remove temporary columns used to compute derived quantities.

    Args:
        data (pandas.DataFrame): Input dataframe containing temporary columns.

    Returns:
        pandas.DataFrame: Cleaned dataframe after temporary columns are removed.
    """
    del data["had_px"]
    del data["had_py"]
    del data["had_pz"]
    del data["p_had"]
    del data["lep_px"]
    del data["lep_py"]
    del data["lep_pz"]
    del data["p_lep"]
    del data["met_x"]
    del data["met_y"]
    del data["jet_leading_px"]
    del data["jet_leading_py"]
    del data["jet_leading_pz"]
    del data["p_jet_leading"]
    del data["jet_subleading_px"]
    del data["jet_subleading_py"]
    del data["jet_subleading_pz"]
    del data["p_jet_subleading"]

    return data


def DER_data(data):
    """Compute all derived quantities and clean up temporary features.

    Args:
        data (pandas.DataFrame): Clean input dataframe without event identifiers.

    Returns:
        pandas.DataFrame: Dataframe with derived features converted to float32.

    Side effects:
        Modifies `data` in place by adding and removing columns.
    """
    data = calcul_int(data)
    data = f_DER_mass_transverse_met_lep(data)
    data = f_DER_mass_vis(data)
    data = f_DER_pt_h(data)
    data = f_DER_deltaeta_jet_jet(data)
    data = f_DER_mass_jet_jet(data)
    data = f_DER_prodeta_jet_jet(data)
    data = f_DER_deltar_had_lep(data)
    data = f_DER_pt_tot(data)
    data = f_DER_sum_pt(data)
    data = f_DER_pt_ratio_lep_had(data)
    data = f_DER_met_phi_centrality(data)
    data = f_DER_lep_eta_centrality(data)
    data = f_del_DER(data)

    logger.debug("Derived Quantities calculated successfully")

    buffer = io.StringIO()
    data.info(buf=buffer, memory_usage="deep", verbose=False)
    info_str = "Data with Derived Quantities :\n" + buffer.getvalue()
    logger.debug(info_str)

    double_precision_cols = data.select_dtypes(include=["float64"]).columns

    logger.debug(f"Converting columns {double_precision_cols} to float32")
    data[double_precision_cols] = data[double_precision_cols].astype(np.float32)
    buffer = io.StringIO()
    data.info(buf=buffer, memory_usage="deep", verbose=False)
    info_str = "Data with Derived Quantities float32 :\n" + buffer.getvalue()
    logger.debug(info_str)
    return data
