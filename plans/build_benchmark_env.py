import os
import shutil
import subprocess
import urllib.request
import zipfile

# Define paths
src_repos = "/project/OmicsDRP_Review/benchmark_sources/repos"
benchmark_dir = "/project/OmicsDRP_Review/Benchmark"
our_data_dir = os.path.join(benchmark_dir, "our_data")

os.makedirs(benchmark_dir, exist_ok=True)
os.makedirs(our_data_dir, exist_ok=True)

# 1. Copy our own preprocessed datasets for easy access
shutil.copy("/project/OmicsDRP_Review/data/IC50_GDSC2.csv", our_data_dir)
shutil.copy("/project/OmicsDRP_Review/data/TargetDrugs_with_MorganFingerprint_GDSC2_512.txt", our_data_dir)
shutil.copy("/project/OmicsDRP_Review/data/Cell_line_meta.csv", our_data_dir)
shutil.copy("/project/OmicsDRP_Review/data/gene_list.txt", our_data_dir)
if os.path.exists("/project/OmicsDRP_Review/data/PGKB_Gene_data_dict.pth"):
    shutil.copy("/project/OmicsDRP_Review/data/PGKB_Gene_data_dict.pth", our_data_dir)

# 2. Copy the 3 selected models to the Benchmark directory
models = ["GraphDRP", "DeepTTA", "PaccMann"]
for model in models:
    src = os.path.join(src_repos, model)
    dest = os.path.join(benchmark_dir, model)
    if os.path.exists(dest):
        print(f"{model} already copied to Benchmark, skipping.")
    else:
        print(f"Copying {model} from repos to Benchmark...")
        shutil.copytree(src, dest)
        print(f"Copied {model} successfully.")

# 3. Download the missing Cell_line_RMA_proc_basalExp.txt for DeepTTA
deeptta_gdsc = os.path.join(benchmark_dir, "DeepTTA", "GDSC_data")
zip_dest = os.path.join(deeptta_gdsc, "Cell_line_RMA_proc_basalExp.txt.zip")
txt_dest = os.path.join(deeptta_gdsc, "Cell_line_RMA_proc_basalExp.txt")

if os.path.exists(txt_dest):
    print("Cell_line_RMA_proc_basalExp.txt already exists, skipping download.")
else:
    url = "http://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip"
    print(f"Downloading cell line basal expression data from {url}...")
    try:
        # Use curl or urllib to download
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(zip_dest, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Downloaded zip successfully. Unzipping...")
        with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
            zip_ref.extractall(deeptta_gdsc)
        print("Unzipped basal expression data successfully.")
        # Remove zip file
        os.remove(zip_dest)
    except Exception as e:
        print(f"Failed to download/unzip basal expression: {e}")
        # Try alternate curl
        try:
            print("Trying curl instead...")
            subprocess.run(["curl", "-o", zip_dest, url], check=True)
            with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
                zip_ref.extractall(deeptta_gdsc)
            os.remove(zip_dest)
            print("Successfully downloaded and unzipped via curl.")
        except Exception as e2:
            print(f"Failed via curl as well: {e2}")

print("Benchmark environment setup complete.")
