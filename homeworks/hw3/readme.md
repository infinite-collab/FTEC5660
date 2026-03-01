# HW3 Submission

This folder contains my **HW3 report** and the **15-minute presentation video**.

The reproduction and evaluation code is hosted in the forked repository: [infinite-collab/AgentGroup](https://github.com/infinite-collab/AgentGroup). The main benchmark script I implemented and modified is `scripts/reproduce_table3_openai.py`.

## How to run the code

Clone the code repository, install dependencies, set your API key and endpoint, and run the benchmark script:

```bash
git clone https://github.com/infinite-collab/AgentGroup.git
cd AgentGroup
pip install -r requirements.txt
```

Then set `OPENAI_API_KEY` and `OPENAI_BASE_URL` as environment variables (or edit them in `scripts/reproduce_table3_openai.py`) and run:

```bash
python scripts/reproduce_table3_openai.py
```

This script evaluates the models configured in the file (currently `gpt-5.2`, `gpt-4-turbo`, and `gpt-3.5-turbo`) and prints the Table-3-style scores.
