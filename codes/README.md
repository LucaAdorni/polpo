# Code Structure

### 1_pre_covid_scrape.py

Function to scrape tweets from January till February 2020. It uses the module *snscrape*, [link](https://github.com/JustAnotherArchivist/snscrape).

### 2_URL_extraction.py

Function to extract all the links shared in our dataset of tweets.

### 3_URL_unshortening_mpi.py

Function to iteratively unshorten the links previously found. The code is built to run on a SLURM server using MPI4py. The companion batch file is **3_URL_unshortening.qsub**.