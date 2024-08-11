# Flickr Image Metadata Updater

This Python script automates the process of updating Flickr image metadata using OpenAI's GPT-4 vision model. It analyzes images in Flickr photosets, generates descriptive titles, detailed descriptions, and relevant keywords, then updates the Flickr metadata accordingly.

## Features

- OAuth authentication with Flickr API
- Retrieval of all photosets or processing of a specific photoset
- Image analysis using OpenAI's GPT-4 vision model
- Automatic updating of Flickr image tags, titles, and descriptions
- Cost tracking for OpenAI API usage

## Prerequisites

- Python 3.x
- Flickr API key and secret
- OpenAI API key

## Installation

1. Clone this repository or download the script.
2. Install the required Python packages: `pip install flickrapi openai python-dotenv`
3. Create a `.env` file in the same directory as the script with the following content:

```
FLICKR_API_KEY=your_flickr_api_key
FLICKR_API_SECRET=your_flickr_api_secret
OPENAI_API_KEY=your_openai_api_key
FLICKR_PHOTOSET_ID=optional_specific_photoset_id
```


## Usage

Run the script using Python: `python flickr_metadata_updater.py`

The script will:
1. Authenticate with Flickr using OAuth
2. Retrieve photosets (all or a specific one)
3. Process each image in the photosets
4. Analyze images using OpenAI's GPT-4 vision model
5. Update Flickr metadata with the generated information
6. Track and display the cost of OpenAI API usage

## Configuration

- `OPENAI_MODEL`: Specifies the OpenAI model to use (default: "gpt-4o-2024-08-06")
- `OPENAI_COST_PER_PROMPT_TOKEN`: Cost per prompt token (default: 0.00250)
- `OPENAI_COST_PER_COMPLETION_TOKEN`: Cost per completion token (default: 0.01000)
- `OPENAI_VISION_COST_PER_IMAGE`: Cost per image analysis (default: 0.000213)

## Notes

- The script skips photosets with names starting with "#" or "@"
- Images that already have descriptions are skipped
- The script handles rate limiting and retries on errors
