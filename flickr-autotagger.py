import flickrapi
from openai import OpenAI, BadRequestError
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv is not installed. Skipping .env file loading.")
except Exception as e:
    print(f"Failed to load .env file: {e}")

# OpenAI Model and costings
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06")
OPENAI_COST_PER_1K_PROMPT_TOKEN = float(os.environ.get("OPENAI_COST_PER_1K_PROMPT_TOKEN", "0.00250"))
OPENAI_COST_PER_1K_COMPLETION_TOKEN = float(os.environ.get("OPENAI_COST_PER_1K_COMPLETION_TOKEN", "0.01000"))
OPENAI_VISION_COST_PER_IMAGE = float(os.environ.get("OPENAI_VISION_COST_PER_IMAGE", "0.000213"))

# Script configuration
FLICKR_PRIVACY_FILTER = int(os.environ.get("FLICKR_PRIVACY_FILTER", "1"))  # 0. none, 1. public, 2. friends, 3. family, 4. friends & family, 5. private
FLICKR_TOKEN_FILE = os.environ.get("FLICKR_TOKEN_FILE", "flickr_token.json")
DESCRIPTIONS_TO_ANALYZE = os.environ.get("DESCRIPTIONS_TO_ANALYZE", '["OLYMPUS DIGITAL CAMERA", "Untitled", "DSC_", "IMG_", "DCIM"]')
DESCRIPTIONS_TO_ANALYZE = eval(DESCRIPTIONS_TO_ANALYZE)
SKIP_PREFIX = os.environ.get("SKIP_PREFIX", '["#", "@"]')
SKIP_PREFIX = eval(SKIP_PREFIX)
MAX_KEYWORDS = int(os.environ.get("MAX_KEYWORDS", "10"))
UPDATED_METADATA_FILE = os.environ.get("UPDATED_METADATA_FILE", "updated_metadata.json")
SINGLE_PHOTOSET_ID = os.environ.get("FLICKR_PHOTOSET_ID")

# Mandatory API environment variables
flickr_api_key = os.environ.get("FLICKR_API_KEY")
flickr_api_secret = os.environ.get("FLICKR_API_SECRET")
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if mandatory API environment variables are set
if not all([flickr_api_key, flickr_api_secret, openai_api_key]):
    print("Please set the required environment variables:")
    print("- FLICKR_API_KEY")
    print("- FLICKR_API_SECRET")
    print("- OPENAI_API_KEY")
    exit(1)

def flickr_authentication():
    try:
        # Check if the token file exists
        if os.path.exists("flickr_token.json"):
            # Load the token from the file
            with open("flickr_token.json", "r") as f:
                try:
                    token_dict = json.load(f)
                    token = flickrapi.auth.FlickrAccessToken(
                        token_dict["oauth_token"],
                        token_dict["oauth_token_secret"],
                        token_dict["access_level"],
                        token_dict["fullname"],
                        token_dict["username"],
                        token_dict["user_nsid"],
                    )
                    flickr_api = flickrapi.FlickrAPI(
                        flickr_api_key, flickr_api_secret, token=token, format="parsed-json"
                    )
                    return flickr_api
                except json.JSONDecodeError:
                    print("Invalid JSON format in the token file. Performing OAuth authentication...")
        
        # Check if the token is available in the environment variable
        elif os.environ.get("FLICKR_OAUTH_TOKEN"):
            token_dict = json.loads(os.environ["FLICKR_OAUTH_TOKEN"])
            token = flickrapi.auth.FlickrAccessToken(
                token_dict["oauth_token"],
                token_dict["oauth_token_secret"],
                token_dict["access_level"],
                token_dict["fullname"],
                token_dict["username"],
                token_dict["user_nsid"],
            )
            flickr_api = flickrapi.FlickrAPI(
                flickr_api_key, flickr_api_secret, token=token, format="parsed-json"
            )
            return flickr_api

        # If the token file doesn't exist and the environment variable is not set, perform the OAuth flow
        flickr_api = flickrapi.FlickrAPI(
            flickr_api_key, flickr_api_secret, format="parsed-json"
        )
        flickr_api.token_cache.forget()
        print("Performing OAuth authentication...")
        flickr_api.get_request_token(oauth_callback="oob")
        authorize_url = flickr_api.auth_url(perms="write")
        print(f"Please visit this URL to authorize the application: {authorize_url}")
        verifier = input("Enter the verifier code: ")
        flickr_api.get_access_token(verifier)

        # Save the full token object as a dictionary
        token_dict = {
            "oauth_token": flickr_api.token_cache.token.token,
            "oauth_token_secret": flickr_api.token_cache.token.token_secret,
            "access_level": flickr_api.token_cache.token.access_level,
            "fullname": flickr_api.token_cache.token.fullname,
            "username": flickr_api.token_cache.token.username,
            "user_nsid": flickr_api.token_cache.token.user_nsid,
        }

        # Save the token dictionary to the file for future use
        with open("flickr_token.json", "w") as f:
            json.dump(token_dict, f)

        return flickr_api
    except flickrapi.FlickrError:
        print("Unauthorized error occurred. Retrying authentication...")
        flickr_api.token_cache.forget()
        return flickr_authentication()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise  # Re-raise the non-FlickrError exception


def get_all_photosets(flickr):
    photosets = []
    page = 1
    per_page = 500

    while True:
        response = flickr.photosets.getList(
            page=page, per_page=per_page, extras="description"
        )
        photosets.extend(response["photosets"]["photoset"])

        if page * per_page >= int(response["photosets"]["total"]):
            break

        page += 1

    return photosets


def has_flickr_description(photo):
    existing_description = photo["description"]["_content"].strip()
    return (
        not any(
            existing_description.startswith(desc) for desc in DESCRIPTIONS_TO_ANALYZE
        )
        and existing_description != ""
    )


def strip_markdown_response(response):
    if response.startswith("```") and response.endswith("```"):
        stripped_response = response[3:-3].strip()
        if stripped_response.startswith("json"):
            stripped_response = stripped_response[4:].strip()
        return stripped_response
    return response


# Function to get image analysis from ChatGPT
def get_image_analysis(
    openai, image_url, photoset_title=None, photoset_description=None, location=None
):
    system_message = (
        "Summarize images and generate metadata as valid JSON. Create JSON with:\n"
        '1. "title": Concise, descriptive title.\n'
        '2. "description": Detailed description with locations, objects, mood, and colors.\n'
        f'3. "keywords": Up to {MAX_KEYWORDS} lowercase keywords, do not include spaces or symbols.\n'
        "Use only the image and optional context. Don't mention context. Keep description/title self-contained.\n"
        "Optional context:\n"
        "- albumTitle: From album title.\n"
        "- albumDescription: From album description.\n"
        "- location: 'latitude' and 'longitude' for photo location.\n"
        'Example JSON: {"title": "Example Title", "description": "Example description.", "keywords": ["keyword1", "keyword2", "keyword3"]}\n'
    )

    user_message = {}
    if photoset_title:
        user_message["albumTitle"] = photoset_title
    if photoset_description:
        user_message["albumDescription"] = photoset_description
    if location:
        user_message["location"] = location

    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(user_message)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=4000,
        )
    except BadRequestError as e:
        raise e

    try:
        analysis = json.loads(
            strip_markdown_response(response.choices[0].message.content.strip())
        )
        if "keywords" in analysis:
            analysis["keywords"] = analysis["keywords"][:MAX_KEYWORDS]

        # Add usage information to the analysis
        if response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = (
                (prompt_tokens * OPENAI_COST_PER_1K_PROMPT_TOKEN / 1000)
                + (completion_tokens * OPENAI_COST_PER_1K_COMPLETION_TOKEN / 1000)
                + OPENAI_VISION_COST_PER_IMAGE
            )
            analysis["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            }

        return analysis
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON response from ChatGPT"}


# Function to update Flickr image tags and description
def update_flickr_metadata(flickr, photo_id, analysis):
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    required_keys = ["title", "description", "keywords"]
    if not all(key in analysis for key in required_keys):
        print(f"Error: Missing required keys in analysis for photo {photo_id}")
        return

    tags = analysis["keywords"]
    title = analysis["title"]
    description = analysis["description"]

    flickr.photos.setTags(photo_id=photo_id, tags=",".join(tags))
    flickr.photos.setMeta(photo_id=photo_id, title=title, description=description)

def process_photoset(flickr, openai, photoset):
    photoset_id = photoset["id"]
    photoset_title = photoset["title"]["_content"]
    photoset_description = photoset["description"]["_content"]

    # Skip photosets with names starting with any character in skip_characters
    if any(photoset_title.startswith(char) for char in SKIP_PREFIX):
        return None

    # Get photos from the current photoset
    try:
        photos = flickr.photosets.getPhotos(
            photoset_id=photoset_id,
            extras="url_m,description,geo",
            media="photos",
            privacy_filter=FLICKR_PRIVACY_FILTER,
        )
    except flickrapi.exceptions.FlickrError as e:
        print(f"Error retrieving photos for photoset: {photoset_title} (ID: {photoset_id}) - {str(e)}")
        return None

    # Check if the photoset has any matching photos, if skipped here check privacy filter
    if photos["photoset"]["total"] == 0:
        return None

    print(f"Processing photoset: {photoset_title} (ID: {photoset_id}) - {photos["photoset"]["total"]} images")
    
    # Process each image in the photoset
    updated_metadata = []
    photoset_cost = 0
    skipped_photos = 0
    for photo in photos["photoset"]["photo"]:
        photo_id = photo["id"]
        image_url = photo["url_m"]

        # Check if the photo already has a description
        if has_flickr_description(photo):
            skipped_photos += 1
            continue

        # Get the location information for the photo
        latitude = photo.get("latitude")
        longitude = photo.get("longitude")
        location = (
            {"latitude": latitude, "longitude": longitude}
            if latitude and longitude
            else None
        )

        retry_count = 0
        while retry_count < 2:
            try:
                # Get image analysis from ChatGPT
                analysis = get_image_analysis(
                    openai, image_url, photoset_title, photoset_description, location
                )
                break
            except BadRequestError as e:
                retry_count += 1
                if retry_count == 2:
                    print(
                        f"Skipping photo {photo_id} due to repeated BadRequestError: {str(e)}"
                    )
                    continue

        if retry_count == 2:
            continue

        # Include photoset_id and photo_id in the analysis JSON
        analysis["photoset_id"] = photoset_id
        analysis["photo_id"] = photo_id

        # Calculate the cost for this request if usage information is available
        if "usage" in analysis:
            if "cost" in analysis["usage"]:
                photoset_cost += analysis["usage"]["cost"]

        # Update Flickr image metadata
        update_flickr_metadata(flickr, photo_id, analysis)

        # Append the analysis result to the updated metadata list
        updated_metadata.append(analysis)

    print(f"Finished processing photoset: {photoset_title}")
    print(f"Skipped {skipped_photos} photos due to existing descriptions")
    print(f"Total cost for photoset: ${photoset_cost:.4f}\n")

    return updated_metadata, photoset_cost

def process_all_photosets(flickr, openai):
    if SINGLE_PHOTOSET_ID:
        # Process a specific photoset
        photosets = [flickr.photosets.getInfo(photoset_id=SINGLE_PHOTOSET_ID)["photoset"]]
    else:
        # Get all photosets
        photosets = get_all_photosets(flickr)

    total_cost = 0
    all_updated_metadata = []
    for photoset in photosets:
        result = process_photoset(flickr, openai, photoset)
        if result is not None:
            updated_metadata, photoset_cost = result
            all_updated_metadata.extend(updated_metadata)
            total_cost += photoset_cost

    return all_updated_metadata, total_cost

def main():
    # Perform OAuth authentication and get the Flickr API instance
    flickr = flickr_authentication()

    # Create OpenAI API client
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_updated_metadata, total_cost = process_all_photosets(flickr, openai)

    # Write the updated metadata JSON to the file only if there are entries
    if all_updated_metadata:
        with open(UPDATED_METADATA_FILE, "w") as f:
            json.dump(all_updated_metadata, f, indent=2)
        print(f"Updated metadata JSON saved to: {UPDATED_METADATA_FILE}")
    else:
        print("No updated metadata to write.")

    print(f"Total cost of OpenAI API usage: ${total_cost:.4f}")

if __name__ == "__main__":
    main()