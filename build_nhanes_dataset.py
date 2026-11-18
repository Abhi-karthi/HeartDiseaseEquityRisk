import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nhanes_raw")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nhanes_cardio_sdoh.csv")


def load_cycle(prefix):
    """Loads and merges a single CDC NHANES cycle from raw SAS transport (.xpt) files."""
    if prefix == "P_":
        demo = pd.read_sas(os.path.join(RAW_DIR, "P_DEMO.xpt"))
        bpx = pd.read_sas(os.path.join(RAW_DIR, "P_BPXO.xpt"))
        bpx["SysBP_raw"] = bpx[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
        bpx["DiaBP_raw"] = bpx[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
        bpq = pd.read_sas(os.path.join(RAW_DIR, "P_BPQ.xpt"))
        tchol = pd.read_sas(os.path.join(RAW_DIR, "P_TCHOL.xpt"))
        diq = pd.read_sas(os.path.join(RAW_DIR, "P_DIQ.xpt"))
        smq = pd.read_sas(os.path.join(RAW_DIR, "P_SMQ.xpt"))
        mcq = pd.read_sas(os.path.join(RAW_DIR, "P_MCQ.xpt"))
        kiq = pd.read_sas(os.path.join(RAW_DIR, "P_KIQ_U.xpt"))
        hiq = pd.read_sas(os.path.join(RAW_DIR, "P_HIQ.xpt"))
        fsq = pd.read_sas(os.path.join(RAW_DIR, "P_FSQ.xpt"))
        dbq = pd.read_sas(os.path.join(RAW_DIR, "P_DBQ.xpt"))
        bmx = pd.read_sas(os.path.join(RAW_DIR, "P_BMX.xpt"))
    else:
        demo = pd.read_sas(os.path.join(RAW_DIR, "DEMO_I.xpt"))
        bpx = pd.read_sas(os.path.join(RAW_DIR, "BPX_I.xpt"))
        bpx["SysBP_raw"] = bpx[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
        bpx["DiaBP_raw"] = bpx[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
        bpq = pd.read_sas(os.path.join(RAW_DIR, "BPQ_I.xpt"))
        tchol = pd.read_sas(os.path.join(RAW_DIR, "TCHOL_I.xpt"))
        diq = pd.read_sas(os.path.join(RAW_DIR, "DIQ_I.xpt"))
        smq = pd.read_sas(os.path.join(RAW_DIR, "SMQ_I.xpt"))
        mcq = pd.read_sas(os.path.join(RAW_DIR, "MCQ_I.xpt"))
        kiq = pd.read_sas(os.path.join(RAW_DIR, "KIQ_U_I.xpt"))
        hiq = pd.read_sas(os.path.join(RAW_DIR, "HIQ_I.xpt"))
        fsq = pd.read_sas(os.path.join(RAW_DIR, "FSQ_I.xpt"))
        dbq = pd.read_sas(os.path.join(RAW_DIR, "DBQ_I.xpt"))
        bmx = pd.read_sas(os.path.join(RAW_DIR, "BMX_I.xpt"))

    df = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]].copy()
    tables = [
        (bpx, ["SEQN", "SysBP_raw", "DiaBP_raw"]),
        (bpq, ["SEQN", "BPQ020", "BPQ050A", "BPQ080", "BPQ100D"]),
        (tchol, ["SEQN", "LBXTC"]),
        (diq, ["SEQN", "DIQ010"]),
        (smq, ["SEQN", "SMQ020", "SMQ040"]),
        (mcq, ["SEQN", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ300A", "MCQ220"]),
        (kiq, ["SEQN", "KIQ022"]),
        (hiq, ["SEQN", "HIQ011", "HIQ210"]),
        (fsq, ["SEQN", "FSDHH"]),
        (dbq, ["SEQN", "DBQ700", "DBD900"]),
        (bmx, ["SEQN", "BMXBMI"]),
    ]
    for sub, cols in tables:
        df = df.merge(sub[cols], on="SEQN", how="left")
    return df


def build_dataset():
    raw_df = pd.concat([load_cycle("P_"), load_cycle("_I")], ignore_index=True)

    # Filter to adults aged 20+ who answered the cardiovascular disease questionnaire
    raw_df = raw_df[raw_df["RIDAGEYR"] >= 20].copy()
    valid_target = raw_df[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"]].isin([1, 2]).any(axis=1)
    raw_df = raw_df[valid_target].copy()

    out = pd.DataFrame(index=raw_df.index)

    # Target: Confirmed Coronary Heart Disease, Myocardial Infarction, Angina, or CHF
    out["HeartDiseaseorAttack"] = (
        (raw_df["MCQ160B"] == 1)
        | (raw_df["MCQ160C"] == 1)
        | (raw_df["MCQ160D"] == 1)
        | (raw_df["MCQ160E"] == 1)
    ).astype(float)

    # 1. Demographics & Continuous Vitals
    out["Age"] = raw_df["RIDAGEYR"].astype(float)
    out["Sex"] = (raw_df["RIAGENDR"] == 1).astype(float)  # 1 = Male, 0 = Female

    # Systolic BP: Adjust by +15 mmHg if taking prescribed antihypertensive meds (BPQ050A == 1)
    # to recover untreated arterial blood pressure load, then impute missing with age/hypertension group median
    sysbp = raw_df["SysBP_raw"].copy()
    sysbp = np.where(raw_df["BPQ050A"] == 1, sysbp + 15.0, sysbp)
    out["SysBP"] = pd.Series(sysbp, index=raw_df.index)
    out["Hypertension"] = ((raw_df["BPQ020"] == 1) | (out["SysBP"] >= 130)).astype(float)
    out["SysBP"] = out["SysBP"].fillna(out.groupby("Hypertension")["SysBP"].transform("median")).round(1)

    # Total Cholesterol: Adjust by 1/0.70 if taking prescribed lipid-lowering statins (BPQ100D == 1)
    # to recover untreated baseline cholesterol level, then impute missing with high-chol group median
    chol = raw_df["LBXTC"].copy()
    chol = np.where(raw_df["BPQ100D"] == 1, chol / 0.70, chol)
    out["TotalChol"] = pd.Series(chol, index=raw_df.index)
    out["HighChol"] = ((raw_df["BPQ080"] == 1) | (out["TotalChol"] >= 200)).astype(float)
    out["TotalChol"] = out["TotalChol"].fillna(out.groupby("HighChol")["TotalChol"].transform("median")).round(1)

    # BMI: Continuous Body Mass Index
    out["BMI"] = raw_df["BMXBMI"].fillna(raw_df["BMXBMI"].median()).round(1)

    # 2. Clinical & Substance History
    # Diabetes: 1.0 = Yes (DIQ010 == 1), 0.5 = Borderline/Prediabetes (DIQ010 == 3), 0.0 = No
    out["Diabetes"] = np.where(raw_df["DIQ010"] == 1, 1.0, np.where(raw_df["DIQ010"] == 3, 0.5, 0.0))

    # Smoker: 0 = Never, 1 = Former (SMQ020 == 1 & SMQ040 == 3), 2 = Current (SMQ020 == 1 & SMQ040 in [1, 2])
    out["Smoker"] = np.where(
        raw_df["SMQ020"] == 1,
        np.where(raw_df["SMQ040"].isin([1, 2]), 2.0, 1.0),
        0.0,
    )

    out["Stroke"] = (raw_df["MCQ160F"] == 1).astype(float)
    out["CKD"] = (raw_df["KIQ022"] == 1).astype(float)
    out["FamilyHistory"] = (raw_df["MCQ300A"] == 1).astype(float)
    out["CancerHistory"] = (raw_df["MCQ220"] == 1).astype(float)

    # 3. Social Determinants of Health (SDOH)
    # IncomeBracket from NHANES Family Income-to-Poverty Ratio (INDFMPIR: 0.0 to 5.0):
    # 1: <$25k (PIR < 1.0), 2: $25k-$50k (1.0-2.0), 3: $50k-$75k (2.0-3.0), 4: $75k-$100k (3.0-4.0), 5: >$100k (>= 4.0)
    pir = raw_df["INDFMPIR"].fillna(raw_df["INDFMPIR"].median())
    out["IncomeBracket"] = np.select(
        [pir < 1.0, pir < 2.0, pir < 3.0, pir < 4.0],
        [1.0, 2.0, 3.0, 4.0],
        default=5.0,
    )

    # HealthcareAccess: 0 = Fully Insured, 1 = Underinsured (coverage gap in past year HIQ210 == 1), 2 = Uninsured (HIQ011 == 2)
    out["HealthcareAccess"] = np.where(
        raw_df["HIQ011"] == 2,
        2.0,
        np.where(raw_df["HIQ210"] == 1, 1.0, 0.0),
    )

    # FoodInsecurity: 1 = USDA Low or Very Low Food Security (FSDHH in [3, 4]), 0 = Full/Marginal Food Security
    out["FoodInsecurity"] = raw_df["FSDHH"].isin([3, 4]).astype(float)

    # DietQuality: 1 = Excellent to 5 = Poor (from DBQ700 self-rated diet quality, default 3 = Good)
    diet = raw_df["DBQ700"].where(raw_df["DBQ700"].isin([1, 2, 3, 4, 5]), 3.0)
    out["DietQuality"] = diet.astype(float)

    # Education: 1 = <9th grade to 5 = College graduate or above (DMDEDUC2)
    edu = raw_df["DMDEDUC2"].where(raw_df["DMDEDUC2"].isin([1, 2, 3, 4, 5]), 3.0)
    out["Education"] = edu.astype(float)

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} with shape {out.shape}")
    print("Columns:", list(out.columns))
    print("Target distribution:\n", out["HeartDiseaseorAttack"].value_counts(normalize=True))


if __name__ == "__main__":
    build_dataset()
