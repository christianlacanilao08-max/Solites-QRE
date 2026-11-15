
# ==============================================================================
# VERSION INFO
# ==============================================================================
__version__ = "1.0.0-beta"
__author__ = "solite"

# ==============================================================================
# PUBLIC API EXPOSURE
# ==============================================================================
from .solites import (
    # Models
    GBM,
    MultiGBM,
    HestonQE,
    HestonEuler,
    
    # Engines
    mc_price,
    rqmc_price,
    mlmc_price,
    simulate_paths,
    
    # Payoffs
    European,
    Digital,
    Asian,
    Barrier,
    Lookback,
    VarianceSwap,
    
    # Risk
    var_cvar,
    var_ci,
    component_cvar,
    spectral_rm,
    evt_var_cvar,
    is_var_linear,
    credit_portfolio,
    kupiec_pof,
    christoffersen,
    
    # Greeks
    greeks_fd,
    greeks_pathwise_lrm,
    
    # Samplers & Special Functions
    norm_cdf,
    norm_ppf,
    norm_pdf,
    erfc,
    bisect_monotone,
    golden_min,
    batch_se,
    
    # Low-Discrepancy Sequences
    halton,
    lattice_cbc,
    PointStream,
    
    # Wiener Path Constructions
    wiener_path,
    construction_spectrum,
    
    # Closed Forms
    bs_price,
    bs_greeks,
    heston_cf_price,
    geo_asian_price,
    barrier_do_call,
    crr_american,
    
    # Jump Diffusion
    merton_mc_price,
    merton_series_price,
    
    # Physics / Feynman-Kac
    dmc_ground_state,
    first_passage_bm,
    
    # Utils
    Grid,
)

# ==============================================================================
# CLEANUP
# ==============================================================================
__all__ = [
    "__version__", "__author__",
    # Models
    "GBM", "MultiGBM", "HestonQE", "HestonEuler",
    # Engines
    "mc_price", "rqmc_price", "mlmc_price", "simulate_paths",
    # Payoffs
    "European", "Digital", "Asian", "Barrier", "Lookback", "VarianceSwap",
    # Risk
    "var_cvar", "var_ci", "component_cvar", "spectral_rm",
    "evt_var_cvar", "is_var_linear", "credit_portfolio",
    "kupiec_pof", "christoffersen",
    # Greeks
    "greeks_fd", "greeks_pathwise_lrm",
    # Samplers & Special Functions
    "norm_cdf", "norm_ppf", "norm_pdf", "erfc",
    "bisect_monotone", "golden_min", "batch_se",
    # Low-Discrepancy Sequences
    "halton", "lattice_cbc", "PointStream",
    # Wiener Path Constructions
    "wiener_path", "construction_spectrum",
    # Closed Forms
    "bs_price", "bs_greeks", "heston_cf_price",
    "geo_asian_price", "barrier_do_call", "crr_american",
    # Jump Diffusion
    "merton_mc_price", "merton_series_price",
    # Physics / Feynman-Kac
    "dmc_ground_state", "first_passage_bm",
    # Utils
    "Grid",
]
