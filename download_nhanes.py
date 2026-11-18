import os
import urllib.request

RAW_DIR = "/Users/aaavi/EquityScoreProject/nhanes_raw"
os.makedirs(RAW_DIR, exist_ok=True)

FILES = {
    # 2017 - March 2020 Pre-Pandemic
    "P_DEMO.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.xpt",
    "P_BPXO.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPXO.xpt",
    "P_BPQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.xpt",
    "P_TCHOL.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_TCHOL.xpt",
    "P_DIQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DIQ.xpt",
    "P_SMQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SMQ.xpt",
    "P_MCQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_MCQ.xpt",
    "P_KIQ_U.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_KIQ_U.xpt",
    "P_HIQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_HIQ.xpt",
    "P_FSQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_FSQ.xpt",
    "P_DBQ.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DBQ.xpt",
    "P_BMX.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.xpt",
    # 2015 - 2016
    "DEMO_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt",
    "BPX_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPX_I.xpt",
    "BPQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPQ_I.xpt",
    "TCHOL_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/TCHOL_I.xpt",
    "DIQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DIQ_I.xpt",
    "SMQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SMQ_I.xpt",
    "MCQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/MCQ_I.xpt",
    "KIQ_U_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/KIQ_U_I.xpt",
    "HIQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HIQ_I.xpt",
    "FSQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/FSQ_I.xpt",
    "DBQ_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DBQ_I.xpt",
    "BMX_I.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.xpt",
}

for fname, url in FILES.items():
    dest = os.path.join(RAW_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"Already exists: {fname}")
        continue
    print(f"Downloading {fname}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as out:
        out.write(resp.read())
    print(f"Saved {fname} ({os.path.getsize(dest)} bytes)")
print("All downloads complete.")
