cat > README.txt << 'EOF'
HFR-TokenSHAP — Feature Importance Estimation Demo
=================================================

This repository contains the demo code and LaTeX paper draft for:
"HFR-TokenSHAP: Hierarchically Restricted Feature TokenSHAP for Binary Phenotype Classification with Large Language Models"

Requested local path reference (original location on the author machine):
/Users/stijnvanseveren/PythonProjects/HFR_TokenSHAP/feature_importance_estimation/demonstration_results/visuals/example_prompt_with_FIEs.png

NOTE: In the GitHub repo, use the *relative* path instead:
demonstration_results/visuals/example_prompt_with_FIEs.png

Overlay description
-------------------
The example_prompt_with_FIEs.png image contains an overlay visualization showing IG and HFR-TokenSHAP
weighted feature-importance (FI) scores on top of an example pseudo-profile prompt. Clinically relevant
features (sleep_quality, childhood_trauma_exposure) are emphasized relative to distractors.

How to run the demo
-------------------
1) Create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

2) Install dependencies
   pip install -r requirements.txt

3) Run the demo
   python demo_run.py

Notes:
- If you hit Hugging Face cache / disk issues, you can clear:
  rm -rf ~/.cache/huggingface

Code pointers
-------------
- demo runner: demo_run.py
- Monte Carlo Shapley (grouped / restricted): utils/group_shapley.py
- Integrated Gradients utilities: utils/ig_attribution.py

Outputs
-------
The demo writes tables into a configured TABLES_DIR in demo_run.py.
Make sure TABLES_DIR points to a valid location on your machine.
EOF
