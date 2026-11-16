from huggingface_hub import snapshot_download,  hf_hub_download
import zipfile, glob


# Don't run this on your personal machine dawg

#! PHASE 1 DATA :

hf_hub_download(repo_id="CraftJarvis/minecraft-vlp", 
                filename="mc-qa-250312.jsonl",
                repo_type="dataset", 
                local_dir="./data_phase_one.nosync")

#! PHASE 2 DATA :

snapshot_download(
    repo_id="CraftJarvis/minecraft-vlp",
    repo_type="dataset",
    local_dir="./data_phase_two.nosync", 
    ignore_patterns=["mc-knowledge-valid.jsonl", "mc-qa-250312.jsonl", "hallucination.jsonl", ".gitattributes"])

#! PHASE 3 DATA :

snapshot_download(repo_id="CraftJarvis/minecraft-vla-sft",
    repo_type="dataset",
    local_dir="./data_phase_three.nosync",
    allow_patterns=['data/train*.parquet']
)

snapshot_download(repo_id="CraftJarvis/minecraft-vla-sft",
    repo_type="dataset",
    local_dir="./data_phase_three_test.nosync",
    allow_patterns=['*valid*']
)

# we need to unzip the zips
for z in glob.glob("./data_phase_two.nosync/images/*.zip"):
    zipfile.ZipFile(z).extractall("./data_phase_two.nosync/images")
