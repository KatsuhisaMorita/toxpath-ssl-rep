# import
import os
import sys
from tqdm import tqdm
import datetime

from dotenv import load_dotenv
import pandas as pd
import numpy as np

# load env
load_dotenv()
root = os.getenv("ROOT_PATH")
HDD_TGGATES_DAYS = os.getenv("HDD_TGGATES_DAYS")
HDD_TGGATES_HOURS = os.getenv("HDD_TGGATES_HOURS")

# import
sys.path.append(f"{root}/src")
from tggate.utils import make_patch


def main():
    df_all = pd.read_csv(f"{root}/data/processed/info_fold.csv", index_col=0)
    lst_compounds = list(set(df_all["COMPOUND_NAME"]))
    print(len(lst_compounds))


if __name__ == "__main__":
    load_dotenv()
    print(os.getenv("HDD_TGGATES_DAYS"))
    main()
