import numpy as np
import pandas as pd


def generate_sample_data(n_samples=240, seed=42):
    rng = np.random.default_rng(seed)

    rainfall_mm = rng.uniform(5, 80, n_samples)
    storm_duration_hr = rng.uniform(0.5, 8.0, n_samples)
    impervious_frac = rng.uniform(0.2, 0.95, n_samples)
    catchment_area_ha = rng.uniform(2, 30, n_samples)
    infiltration_index = rng.uniform(5, 40, n_samples)
    slope_pct = rng.uniform(0.2, 5.0, n_samples)

    intensity_mmphr = rainfall_mm / storm_duration_hr
    threshold_term = np.maximum(impervious_frac - 0.55, 0.0) ** 2

    noise = rng.normal(0, 4.0, n_samples)

    peak_flow_cms = (
        1.8
        + 0.055 * rainfall_mm
        + 0.9 * intensity_mmphr
        + 7.5 * impervious_frac
        + 0.018 * rainfall_mm * impervious_frac
        + 6.0 * threshold_term
        - 0.035 * infiltration_index
        - 0.015 * infiltration_index * np.sqrt(rainfall_mm)
        + 0.012 * catchment_area_ha * slope_pct
        + noise
    )

    time_to_peak_min = (
        140
        - 1.2 * intensity_mmphr
        - 18 * impervious_frac
        + 0.7 * infiltration_index
        + rng.normal(0, 6, n_samples)
    )

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
    df = generate_sample_data()
    df.to_csv("../data/raw_data/stormwater_events_sample.csv", index=False)
    print("Saved raw sample data.")


if __name__ == "__main__":
    main()