import os
import subprocess
import urllib.request
import urllib.error

# Setup folders
base_dir = "/project/OmicsDRP_Review/benchmark_sources"
repos_dir = os.path.join(base_dir, "repos")
papers_dir = os.path.join(base_dir, "papers")

os.makedirs(repos_dir, exist_ok=True)
os.makedirs(papers_dir, exist_ok=True)

repos = {
    "PaccMann": "https://github.com/PaccMann/paccmann_predictor",
    "GraphDRP": "https://github.com/hauldhut/GraphDRP",
    "DeepTTA": "https://github.com/jianglikun/DeepTTC",
    "DeepDRA": "https://github.com/t-m-vardin/DeepDRA",
    "CSG2A": "https://github.com/eugenebang/CSG2A",
    "TGSA": "https://github.com/violet-sto/TGSA",
    "DeepCDR": "https://github.com/kimmo1019/DeepCDR",
    "DrugCell": "https://github.com/idekerlab/DrugCell",
    "SWnet": "https://github.com/zuozhaorui/SWnet",
    "DRPreter": "https://github.com/babaling/DRPreter",
    "GPDRP": "https://github.com/yyk124/GPDRP"
}

papers = {
    "PaccMann": "https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520",
    "GraphDRP": "https://ieeexplore.ieee.org/document/9359501",
    "DeepTTA": "https://academic.oup.com/bib/article/23/3/bbac100/6554594?login=true",
    "DeepDRA": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307649",
    "CSG2A": "https://academic.oup.com/bioinformatics/article/40/Supplement_1/i130/7700850",
    "TGSA": "https://academic.oup.com/bioinformatics/article/38/2/496/6380567",
    "DeepCDR": "https://academic.oup.com/bioinformatics/article/36/Supplement_2/i911/6055929",
    "DrugCell": "https://www.sciencedirect.com/science/article/pii/S1535610820304888",
    "SWnet": "https://link.springer.com/article/10.1186/s12859-021-04352-9",
    "DRPreter": "https://www.mdpi.com/1422-0067/23/22/13919",
    "GPDRP": "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05618-0"
}

# Clone repositories
for name, url in repos.items():
    dest = os.path.join(repos_dir, name)
    if os.path.exists(dest):
        print(f"{name} already exists, skipping clone.")
        continue
    print(f"Cloning {name} from {url}...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
        print(f"Successfully cloned {name}.")
    except Exception as e:
        print(f"Failed to clone {name}: {e}")

# Fetch papers (HTML structure / abstract page)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
for name, url in papers.items():
    dest = os.path.join(papers_dir, f"{name}_paper.html")
    if os.path.exists(dest):
        print(f"{name} paper already exists, skipping download.")
        continue
    print(f"Downloading paper for {name} from {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read()
            with open(dest, 'wb') as f:
                f.write(html)
        print(f"Successfully downloaded paper for {name}.")
    except urllib.error.URLError as e:
        print(f"Failed to download paper for {name}: {e}")
    except Exception as e:
        print(f"Failed to download paper for {name}: {e}")
