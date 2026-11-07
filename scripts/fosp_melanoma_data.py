"""Load the FOSP cohort with the existing preprocessor's extended follow-up."""

from pathlib import Path

import pandas as pd

from scripts.preprocessing import FOSPMelanomaTargetEncodedPreprocessor


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "melanoma_fosp_gomez2023.csv"


def load_fosp_melanoma(data_path=DATA_PATH):
    """Preserve CSV row order and use the same event indicator for all splits.

    ``tempo_ext`` is already in years and ``status_ext`` is already coded 0/1.
    Covariate scaling and survival target encoders are fitted later,
    independently inside each training fold.
    """
    frame = pd.read_csv(data_path).reset_index(drop=True)
    preprocessor = FOSPMelanomaTargetEncodedPreprocessor()
    preprocessor._validate(frame)
    if len(frame) < 2 or frame[preprocessor.status_col].nunique() != 2:
        raise ValueError("FOSP melanoma data must contain both events and censored observations.")
    return frame.assign(delta=frame[preprocessor.status_col].astype(int))
