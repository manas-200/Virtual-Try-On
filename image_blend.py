import os
import base64
import mimetypes
from google import genai
from google.genai import types
from PIL import Image as PILImage

def display_image(image_path, title="Image"):
    print(f"{title}: {image_path}")
    img = PILImage.open(image_path)
    img.show()

def blend_images(person_image_path, clothing_image_path, api_key, prompt_text, output_file_name="blended_output"):
    # Authenticate
    os.environ["GEMINI_API_KEY"] = api_key
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Upload images
    file1 = client.files.upload(file=person_image_path)
    file2 = client.files.upload(file=clothing_image_path)
    
    # Prepare contents
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=file1.uri, mime_type=file1.mime_type),
                types.Part.from_uri(file_uri=file2.uri, mime_type=file2.mime_type),
                types.Part.from_text(text=prompt_text),
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["image", "text"],
        response_mime_type="text/plain",
    )

    model = "gemini-2.0-flash-exp-image-generation"
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        part = chunk.candidates[0].content.parts[0]
        if hasattr(part, "inline_data") and part.inline_data:
            inline_data = part.inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type) or ".png"
            full_output_path = f"{output_file_name}{file_extension}"

            with open(full_output_path, "wb") as f:
                f.write(data_buffer)
            print(f"Output saved to: {full_output_path}")

            display_image(full_output_path, title="Blended Output")

        elif hasattr(part, "text") and part.text:
            print("Text response:")
            print(part.text)

def main():
    API_KEY = "AIzaSyA7zAHjg1Tzafs2P0NIQ4qrhiIqprzHKrE"  
    image1_path = "human.jpg"   # Make sure image is in same directory 
    image2_path = "cloth.jpg"

    prompt_text = "blend the two images in such a way that the man in the first image is wearing the clothes from the second image"
    output_file_name = "blended_output"

    # Show input images
    display_image(image1_path, title="Person Image")
    display_image(image2_path, title="Clothing Image")

    # Call blending function
    blend_images(
        person_image_path=image1_path,
        clothing_image_path=image2_path,
        api_key=API_KEY,
        prompt_text=prompt_text,
        output_file_name=output_file_name
    )

if __name__ == "__main__":
    main()
