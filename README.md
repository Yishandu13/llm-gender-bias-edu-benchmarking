# Benchmark work on llm-gender-implicit-bias-in-education
This repository is under construction and will be used for a benchmark study on "Do LLMs show gender bias when generating feedback for essay writing?"
The structure of this repository is:
```
.
├── notebooks/                 
│   └── main.ipynb             # Main experiment code
│
├── data/                                   
│   ├── essay data/            # data of essay writing (original and counterfactual)
│       ├──original      
│       └──counterfactal       
│   ├── gender words/          # gender words used to label essays
│   ├── prompts/          
│       ├──prompt template.txt     # LLM prompt template used in this study
│       └──prompts used.csv  
│   └── feedback to essays/             # response from LLMs
│
├── data collection pipeline/     # GPT-4o, DeepSeek, LLaMA
│   ├── gender context.py       # construct counterfactual gender essays and save into data
│   ├── construct_prompts.py   # construct prompts based on templates and raw data and save into data
│   ├── query_llms/          # call llms with prompts
│       ├──llm_interface.py       # unified interface management (UIM)
│       ├──gpt_openai.py          # packaging OpenAI GPT-4o calls
│       ├──deepseek_api.py        # packaging DeepSeek API calls
│       └──llama_local.py         # local LLaMA model runs
│   └── postprocess.py         # standardise llms output formats
│  
├── evaluation/                # bias analysis function module
│   ├── cosine_similarities.py # calculate cosine similarities
│   ├── score_value_analysis.py  # analysis of scores given to essays
│   └── bias_metrics.py        # OR, SWEAT, LIWC, t-testing
│
├── config/                    # configuration file (key, path)
│   └── config.yaml
│
├── README.md
└── requirements.txt
```
