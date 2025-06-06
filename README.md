# CFRM521_FinalProject

# -----

# Introduction

Pricing options is essential for financial institutions to assess the fair value and risk associated with these instruments. Accurate pricing supports informed investment decisions and enables effective hedging strategies, while inaccurate pricing can lead to inefficient hedging and potentially severe financial losses. Although option pricing can be complex, even relatively simple models can achieve a high degree of accuracy in certain scenarios. Several well-established models exist for pricing options, including the Black-Scholes model, the Heston model, and Dupire’s local volatility model. More recently, the rise of machine learning has spurred a growing body of research focused on using data-driven methods—such as neural networks, support vector regression, and decision trees—for option pricing. In this paper, we investigate the efficacy of several common machine learning techniques in pricing American call options and demonstrate that even simple models can yield promising results.

## Goal: To predict American Call Option prices

## Data: Historical Options and Stock prices from February 2013
### Original Data Source

The original data for this project is sourced from https://optiondata.org/. They have graciously provided free sample datasets for option prices and stock prices from January 2013 to June 2013. The data we will be using is from the month of February 2013 located in the `2013-02.zip` file located on their website.

### Data Preprocessing

The dataset we will be using is fairly large, there are 19 trading days in the month of February 2013 and two different csv files for each day, the first being the options data and the second being the stock price data. 

To simplify this, we will first join the two datasets through the `symbol` column in the stocks files, and the `underlying` column in the options files. This will allow us to access the data for both in a single file for our features later.

After reviewing the files, we found there were around 500,000 rows for 3800 unique tickers on each trading day, with 19 different days that meant we were looking at close to 9.5 million rows of data. Training machine learning algorithms on this much data will take a very long time, so to minimize this effect, we decided to filter the data to a subset of S&P 100 stocks as of February 1st 2013 and filter the options to call options only, allowing us to train our models within a reasonable time-frame.

Lastly, since our original options data set contained bid and ask prices, we engineered a new feature named `mid_price` which contains the average of the two, and will be evaluated as the price of the option.

Below I've attached a code chunk which displays the filtering we performed. (Please note this won't work on your machine unless you've modified the directories with the location of the files on your own machine):

```
import numpy as np 
import pandas as pd
import os

def combine_options_data(options_data, stock_data):
    options_data['mid_price'] = (options_data['bid'] + options_data['ask'])/2
    combined_df = options_data.merge(stock_data, 
                                     left_on = "underlying", right_on = "symbol",
                                     suffixes=('', '_stock'))
    combined_df = combined_df.drop(columns=['symbol'])
    return combined_df

def combine_new_data(date):
    options = pd.read_csv(f"/home/steve/Downloads/CFRM_521/ProjectData/2013-02/{date}options.csv")
    stocks = pd.read_csv(f"/home/steve/Downloads/CFRM_521/ProjectData/2013-02/{date}stocks.csv")
    return combine_options_data(options, stocks)
        
url = "https://web.archive.org/web/20130201003232/https://en.wikipedia.org/wiki/S%26P_100"
sp100_comp = pd.read_html(url)
sp100_comp = sp100_comp[2]["Symbol"]
sp100_comp = sp100_comp.unique()

dir = "/home/steve/Downloads/CFRM_521/ProjectData/2013-02/"
dates = []
for file in os.listdir(dir):
    #print(file)
    if file.endswith(".csv"):
        if "options" in file:
            dates.append(file.split("options")[0])
    
dates = sorted(set(dates))
for date in dates:
    print(f"Processing: {date}")
    combined_df = combine_new_data(date)
    combined_df['underlying'] = (
    combined_df['underlying']
    .str.replace('.', '-', regex=False)
    .str.upper()
    .replace({'GOOGL': 'GOOG'})
    )
    sp100_comp = [sym.replace('.', '-').upper() for sym in sp100_comp]
    filt_df = combined_df[combined_df['underlying'].isin(sp100_comp)]
    num_stocks = filt_df['underlying'].unique()
    if len(num_stocks) != len(sp100_comp):
        print(f"Error size mismatch, Filtered Df Size: {len(num_stocks)}, SP100 Size: {len(sp100_comp)}")
        missing = set(sp100_comp) - set(filt_df['underlying'].unique())
        print(f"Missing tickers: {sorted(missing)}")
    else:
        filt_df.to_csv(f"/home/steve/Downloads/CFRM_521/ProjectData/filtered/{date}.csv", index = False)
```

### What is an instance in our data?:
An instance (row) of our data would include:
  * The OPRA contract ID (`contract`)
  * Ticker of the underlying asset (`underlying`)
  * Expiration date of the option contract (`expiration`)
  * Option contract type (`type` (filtered to only Call options))
  * Option Strike price (`strike`)
  * Style of option (`style` (i.e American options, European options, etc.))
  * Option bid price (`bid`)
  * Option bid quantity (`bid_size`)
  * Option ask price (`ask`)
  * Option ask quantity (`ask_size`)
  * Quantity of the option traded that day (`volume`)
  * Quantity of the option currently active in the market (`open_interest`)
  * Date of options data being observed (`quote_date`)
  * Option contract's delta (`delta`)
  * Option contract's gamma (`gamma`)
  * Option contract's theta (`theta`)
  * Option contract's vega (`vega`)
  * Option contract's implied volatility (`implied_volatility`)
  * Option contract's mid-price (`mid_price`)
  * Underlying asset's opening price (`open`)
  * Underlying asset's highest price of the day (`high`)
  * Underlying asset's lowest price of the day (`low`)
  * Underlying asset's closing price (`close`)
  * Quantity of the underlying asset traded (`volume_stock`)
  * Underlying asset's adjusted closing price (`close`)

## Target Variable: American Call Option Mid-Price (`mid_price`)

## Features Used:
  * Historical Option Strike Prices (`strike`)
  * Underlying asset price (`close`)
  * Option Greeks (`delta`, `gamma`, `theta`, `vega`)
  * Implied Volatility (`implied_volatility`)
  * Time to Expiry (`tte`)
