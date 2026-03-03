# Emoji Data Project

## Project Overview
This repository contains the research materials, datasets, and analysis scripts for a university data course project. The primary objective of the study was to analyze emoji usage and sentiment trends in movie reviews across Twitter. 

The project evaluates how emotional expression varies by platform using sentiment lexicons to categorize text-based data. This repository serves as a static archive of the project's methodology and results.

## Repository Contents
The repository is organized to include all stages of the data pipeline, from raw acquisition to final reporting:

* **Data Assets**: 
    * `movie_data/`: Contains datasets used for sentiment analysis.
    * `positive-words.txt` & `negative-words.txt`: The sentiment lexicons used to score text data.
* **Analytical Scripts**:
    * `reddit_amazon_dataset.py`: Processes and analyzes sentiment within the Reddit and Amazon datasets.
    * `download.py`: Utility used to fetch the initial raw data.
    * `check.py`: Validation script for data integrity.
* **Visualizations**:
    * `movie_viz/`: Visual output (charts/graphs) generated from movie-related data.
    * `tweet_viz/`: Visual output generated from Twitter data.
    * `sentiments_pie.py`: The script responsible for generating sentiment distribution pie charts.
* **Final Deliverables**:
    * Includes the final project report and supporting documentation summarizing the findings.

## Methodology
The project utilized a Python-based workflow to:
1.  Incorporate multi-platform data into a unified analysis environment.
2.  Apply lexicon-based sentiment analysis to determine the emotional polarity of text.
3.  Generate visual representations of data to identify patterns in emoji distribution and sentiment.
