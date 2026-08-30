# Dataset Documentation

## Overview

This project uses the **News Category Dataset v3** from HuffPost for news topic classification. The dataset contains news articles categorized into 42 topics, providing a rich resource for training and evaluating machine learning models.

## Dataset Details

- **Source**: [HuffPost News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset) (v3)
- **Format**: JSON Lines (one JSON object per line)
- **Total Samples**: 209,527 news articles
- **Features**:
  - `category`: News category (42 unique classes)
  - `headline`: Article headline
  - `authors`: Author(s) of the article
  - `link`: URL to the full article
  - `short_description`: Brief summary
  - `date`: Publication date
- **Class Distribution**: Highly imbalanced (e.g., POLITICS: 16.99%, EDUCATION: 0.48%)
- **Text Statistics**:
  - Headlines: Mean length ~58 chars, max 320 chars
  - Descriptions: Mean length ~114 chars, max 1472 chars
  - Empty descriptions: ~9.4% of samples

## Data Quality

- **Duplicates**: 13 exact duplicates, 489 duplicate text pairs
- **Missing Data**: Minimal, mostly in descriptions
- **Preprocessing**: Basic cleaning applied; embeddings added in v2.0

## Usage

- **Raw Data**: Located in `data/raw/huffpost/News_Category_Dataset_v3.json`
- **Versioning**: Managed with DVC (see `.dvc` file)
- **Quality Checks**: Run `python scripts/data_quality.py data/raw/huffpost/News_Category_Dataset_v3.json`

## Notes

- Dataset is in English only.
- For production use, consider data augmentation or balancing techniques due to class imbalance.
- Last updated: October 2025
