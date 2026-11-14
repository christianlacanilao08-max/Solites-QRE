
__version__ = "1.0.0-beta"
__author__ = "solite"

try:
    from .solites import (
        # Models
        GBM,
        HestonQE,
        MultiGBM,
        MertonJumpDiffusion, # If you split this out
        
        # Engines
        mc_price,
        mlmc_price,
        rqmc_price,
        
        # Payoffs
        European,
        Asian,
        Barrier,
        Lookback,
        Digital,
        
        # Risk
        var_cvar,
        evt_var_cvar,
        credit_portfolio,
        
        # Utils
        Grid,
        norm_ppf,
        bs_price,
    )
except ImportError:
    # Fallback if structure is different
    pass

__all__ = [
    "GBM", "HestonQE", "MultiGBM",
    "mc_price", "mlmc_price", "rqmc_price",
    "European", "Asian", "Barrier", "Lookback", "Digital",
    "var_cvar", "evt_var_cvar", "credit_portfolio",
    "Grid", "norm_ppf", "bs_price",
    "__version__", "__author__"
]
