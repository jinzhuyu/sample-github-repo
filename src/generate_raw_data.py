import numpy as np
import pandas as pd


def generate_sample_data(n_samples=240, seed=42):
    """
    Generate synthetic stormwater event data.

    The goal is to create a dataset that mimics real hydrologic behavior,
    with nonlinear relationships and noise for ML modeling.
    """
    rng = np.random.default_rng(seed)  # reproducible random number generator

    # --------------------------------------------------
    # 1. Generate basic input variables (features)
    # --------------------------------------------------
    rainfall_mm = rng.uniform(5, 80, n_samples)           # total rainfall depth
    storm_duration_hr = rng.uniform(0.5, 8.0, n_samples)  # storm duration
    impervious_frac = rng.uniform(0.2, 0.95, n_samples)   # % impervious surface
    catchment_area_ha = rng.uniform(2, 30, n_samples)     # drainage area
    infiltration_index = rng.uniform(5, 40, n_samples)    # soil infiltration capacity
    slope_pct = rng.uniform(0.2, 5.0, n_samples)          # terrain slope

    # --------------------------------------------------
    # 2. Derived hydrologic variables
    # --------------------------------------------------
    intensity_mmphr = rainfall_mm / storm_duration_hr
    # rainfall intensity = depth / duration (key driver of runoff)

    threshold_term = np.maximum(impervious_frac - 0.55, 0.0) ** 2
    # nonlinear threshold effect:
    # runoff increases rapidly once imperviousness exceeds ~55%

    # --------------------------------------------------
    # 3. Add stochastic noise
    # --------------------------------------------------
    noise = rng.normal(0, 4.0, n_samples)
    # represents measurement error + unobserved processes

    # --------------------------------------------------
    # 4. Generate target variable: peak flow
    # --------------------------------------------------
    peak_flow_cms = (
        1.8
        + 0.055 * rainfall_mm                 # more rainfall → higher flow
        + 0.9 * intensity_mmphr              # higher intensity → higher peak
        + 7.5 * impervious_frac              # impervious area strongly increases runoff
        + 0.018 * rainfall_mm * impervious_frac  # interaction effect
        + 6.0 * threshold_term               # nonlinear urbanization effect
        - 0.035 * infiltration_index         # infiltration reduces runoff
        - 0.015 * infiltration_index * np.sqrt(rainfall_mm)  # interaction
        + 0.012 * catchment_area_ha * slope_pct  # larger + steeper → faster flow
        + noise
    )

    # --------------------------------------------------
    # 5. Secondary target: time to peak
    # --------------------------------------------------
    time_to_peak_min = (
        140
        - 1.2 * intensity_mmphr             # intense storms peak faster
        - 18 * impervious_frac              # urban areas respond quickly
        + 0.7 * infiltration_index          # infiltration delays peak
        + rng.normal(0, 6, n_samples)
    )

    # --------------------------------------------------
    # 6. Assemble dataset
    # --------------------------------------------------
    df = pd.DataFrame(
        {
            "event_id": np.arange(1, n_samples + 1),
            "storm_duration_hr": storm_duration_hr.round(3),
            "rainfall_mm": rainfall_mm.round(3),
            "impervious_frac": impervious_frac.round(3),
            "catchment_area_ha": catchment_area_ha.round(3),
            "infiltration_index": infiltration_index.round(3),
            "slope_pct": slope_pct.round(3),
            "peak_flow_cms": peak_flow_cms.round(3),
            "time_to_peak_min": time_to_peak_min.round(3),
        }
    )

    return df


def main():
    """
    Generate dataset and save to CSV.

    This is the first step in the workflow:
    data generation → data processing → modeling → plotting
    """
    df = generate_sample_data()

    # Save as raw data for downstream scripts
    df.to_csv("../data/raw_data/stormwater_events_sample.csv", index=False)

    print("Saved raw sample data.")


if __name__ == "__main__":
    main()