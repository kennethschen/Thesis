# Senior Thesis, Statistics, Harvard University

As financial research has long sought to capture future stock market movements, quantitative metrics are often exploited as soon as they are discovered and arbitrage opportunities are scarce. As an additional means to predict market movements, text data from news articles offers qualitative information that may contain sentiments and investor biases not captured by quantitative metrics alone on which investors may exploit. The psychological nature of news articles has been examined and tokenized to a degree for daily stock market predictions but not extensively for multistep predictions into the future combined with quantitative metrics. Additionally, the statistics and machine learning communities have generally performed their analysis of stock markets using tools from their respective fields only. While machine learning black box algorithms generally are richer than statistical models, their lack of distributional assumptions can lead to poor fitting and numerical instability. In this paper, a novel foundational multistep time series model is built that leverages statistical ARIMA models to provide numerical stability while leveraging the richness of the LSTM machine learning model on stock time series and news articles data.

* **create_lightweight_database.ipynb**: Creates a SQLite database for AAPL, AMZN, DIS, and GS by scraping Yahoo Finance
* **get_nyt_articles.ipynb**: Calls the NYT API and scrapes NYT articles and stores each revelant article in raw text format as a pickled object
* **get_stock_prices.ipynb**: Creates a SQLite database for all S\&P 500 Companies by scraping Yahoo Finance
* **gpu_check.ipynb**: Checks if GPUs are used for faster computing while on AWS
* **run_models.ipynb**: Runs the models discussed in this paper and produces all plots, making calls to the .py files for defined functions
* **run_models_no_dl_output.ipynb**: Same as the previous notebook, except training verbosity is removed and only non-GPU dependent cells are run.
* **model_functions.py**: Contains all functions to preprocess the data, run the models, and perform forecasts
* **save_load_functions.py**: Contains all functions to save and load models, losses, plots, dataframes, etc.
* **tokenization_functions.py**: Tokenizes the pickled raw news article data

Completed in April, 2019.
