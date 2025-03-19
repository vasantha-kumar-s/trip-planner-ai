from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import base64
import httpx
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import re
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

if torch.cuda.is_available():
    model = model.to("cuda")

# API endpoint for blog generation
@app.post("/generate-blog")
async def generate_blog(
    images: List[UploadFile] = File(...),
    tone: str = Form(...),
    length: str = Form(...),
    audience: str = Form(None),
    topics: str = Form(None)
):
    try:
        # Process all images (up to 5)
        captions = []
        for image in images[:5]:  # Limit to 5 images
            image_content = await image.read()
            pil_image = Image.open(io.BytesIO(image_content))
            
            # Generate image caption using BLIP
            inputs = processor(pil_image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            output = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)
        
        # Extract key themes from captions
        all_captions_text = " ".join(captions)
        themes = extract_themes(all_captions_text)
        
        # Map length to word count
        word_count_map = {
            "short": 300,
            "medium": 600,
            "long": 1000
        }
        target_word_count = word_count_map.get(length, 600)
        
        # Prepare prompt for LLM
        topics_str = f"and focusing on these topics: {topics}" if topics else ""
        audience_str = f"for an audience of {audience}" if audience else "for a general audience"
        
        prompt = f"""
        Generate a creative, narrative-driven blog post inspired by these image captions: "{all_captions_text}"
        
        Tone: {tone}
        Target length: approximately {target_word_count} words
        Target audience: {audience_str}
        {topics_str}
        
        Important guidelines:
        1. Create a COMPELLING NARRATIVE that ties the images together - don't just describe them.
        2. Generate a SHORT, CATCHY TITLE (3-4 words only) that captures the essence of the story.
        3. Use the images as inspiration for storytelling, not just as items to describe.
        4. Focus on creating an emotional journey that uses the images as waypoints.
        5. The blog should tell a coherent story with a beginning, middle, and end.
        
        Key themes to incorporate in the narrative: {', '.join(themes)}
        
        The blog should include:
        1. A concise, impactful title (3-4 words only)
        2. A compelling introduction that hooks the reader
        3. Main content with appropriate subheadings and storytelling elements
        4. A conclusion that ties the narrative together
        
        Also generate:
        - 5-7 SEO keywords
        - 3-5 hashtags for social media
        - Alt text descriptions for each image that fit the narrative
        - Meta description (150-160 characters)
        
        Format the blog with proper HTML tags and structure for web publishing.
        """
        
        # Call external LLM API (Claude or Gemini)
        blog_content, metadata = await call_llm_api(prompt, captions, themes)
        
        # Structure the response
        response = {
            "blog_content": blog_content,
            "keywords": metadata["keywords"],
            "hashtags": metadata["hashtags"],
            "alt_texts": metadata["alt_texts"],
            "meta_description": metadata["meta_description"],
            "title": metadata["title"]
        }
        
        return response
        
    except Exception as e:
        return {"error": str(e)}

def extract_themes(text: str, num_themes: int = 5) -> List[str]:
    """Extract key themes from the text"""
    # Tokenize and clean the text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_freq = Counter(filtered_tokens)
    
    # Get the most common words as themes
    themes = [word for word, _ in word_freq.most_common(num_themes)]
    
    # Add some abstract themes based on common concepts
    abstract_themes = ["connection", "journey", "transformation", "discovery", "memory", 
                      "contrast", "harmony", "emotion", "perspective", "moment", 
                      "reflection", "change", "growth", "balance", "essence"]
    
    # Choose 2-3 random abstract themes
    selected_abstracts = random.sample(abstract_themes, min(3, num_themes - len(themes)))
    themes.extend(selected_abstracts)
    
    return themes[:num_themes]

async def call_llm_api(prompt: str, captions: List[str], themes: List[str]):
    """
    Call an external LLM API (Claude or similar)
    
    For this example, we'll simulate a response instead of making an actual API call
    In a real implementation, you would call your LLM API endpoint
    """
    # This is a simulation - in production, replace with actual API call
    # For example:
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(
    #         "https://api.anthropic.com/v1/messages",
    #         headers={"Authorization": f"Bearer {os.getenv('ANTHROPIC_API_KEY')}"},
    #         json={"model": "claude-3-opus-20240229", "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]}
    #     )
    #     result = response.json()
    #     blog_content = result["content"][0]["text"]
    
    # For demonstration, generate a simulated blog post
    # Extract core concepts from captions
    core_concepts = []
    for caption in captions:
        words = caption.split()
        if len(words) > 3:
            core_concepts.append(words[len(words)//2])
    
    # Generate a concise title (3-4 words)
    title_options = [
        "Beyond The Horizon",
        "Whispers Of Light",
        "Moments In Time",
        "Echoes Of Memory",
        "Colors Of Life",
        "Silent Conversations",
        "Untold Stories",
        "Fragments Of Time"
    ]
    
    # Choose a title based on themes if possible
    if themes:
        primary_theme = themes[0].capitalize()
        secondary_theme = themes[1].capitalize() if len(themes) > 1 else "Journey"
        title = f"{primary_theme} & {secondary_theme}"
        
        # Make sure it's 3-4 words
        if len(title.split()) > 4:
            title = random.choice(title_options)
    else:
        title = random.choice(title_options)
    
    # Generate keywords from themes and captions
    keywords = generate_keywords(" ".join(captions) + " " + " ".join(themes), 7)
    hashtags = generate_hashtags(keywords, 5)
    
    # Generate alt texts for each image that align with the narrative
    alt_texts = []
    for i, caption in enumerate(captions):
        # Create more narrative-focused alt text
        narrative_elements = [
            "A moment of reflection showing",
            "A captivating scene featuring",
            "A powerful visual depicting",
            "An evocative image revealing",
            "A stunning perspective of"
        ]
        
        # Condense the caption
        condensed_caption = caption.split(".")[0] if "." in caption else caption
        if len(condensed_caption.split()) > 10:
            condensed_caption = " ".join(condensed_caption.split()[:10]) + "..."
            
        alt_texts.append(f"{random.choice(narrative_elements)} {condensed_caption.lower()}")
    
    # Ensure we have alt texts for all images
    while len(alt_texts) < len(captions):
        alt_texts.append(f"A key moment in our visual journey")
    
    # Generate meta description
    meta_description = f"Explore {title} - a visual journey through {len(captions)} captivating moments that reveal the beauty of {themes[0]} and the power of {themes[1] if len(themes) > 1 else 'storytelling'}."
    
    # Simulate blog content generation with better storytelling
    blog_content = f"""
    <h1>{title}</h1>
    
    <p>Some stories can't be told with words alone. They require images—moments frozen in time that speak directly to our hearts and imaginations. This visual journey invites you to step beyond the ordinary and experience a narrative that unfolds across {len(captions)} carefully curated frames.</p>
    
    <h2>The Beginning: An Invitation</h2>
    
    <p>Every journey begins with a single step. In this case, our first image serves as a doorway into a world where {themes[0]} and {themes[1] if len(themes) > 1 else 'beauty'} intersect in unexpected ways. What appears at first glance to be a simple scene reveals itself as something more profound upon closer inspection.</p>
    
    <p>The visual language here speaks of possibilities—of paths not yet taken and stories waiting to unfold. It whispers of adventures just beginning and invites us to venture further into the narrative.</p>
    
    <h2>The Heart of the Matter</h2>
    
    <p>As our journey continues, we find ourselves drawn deeper into the story. The images now reveal connections and contrasts that weren't immediately apparent. Light plays against shadow, movement against stillness, creating a visual rhythm that pulls us forward.</p>
    
    <p>Here, in the middle of our story, we discover that these images aren't just documentation—they're transformation. They take ordinary moments and elevate them to something extraordinary, something worthy of our attention and reflection.</p>
    
    <h2>Unexpected Revelations</h2>
    
    <p>The most powerful stories often surprise us. They lead us down familiar paths only to reveal unexpected vistas. Our visual journey is no different. Each image builds upon the last, creating a narrative that's greater than the sum of its parts.</p>
    
    <p>In these frames, we find echoes of our own experiences—moments of {themes[2] if len(themes) > 2 else 'connection'} that resonate with something deep within us. They remind us that storytelling is not just about what we see, but about what we feel and remember.</p>
    
    <h2>The Unseen Thread</h2>
    
    <p>Throughout this visual narrative runs an unseen thread—a connection that binds these separate moments into a cohesive whole. It's the emotional undercurrent that gives meaning to the images, transforming them from mere pictures into a story worth experiencing.</p>
    
    <p>This thread speaks of {themes[0]} and {themes[1] if len(themes) > 1 else 'emotion'}, of the human experience distilled into visual form. It asks us not just to look, but to see—to engage with the narrative on a deeper level.</p>
    
    <h3>Coming Full Circle</h3>
    
    <p>As our journey concludes, we find ourselves changed by what we've witnessed. The final images bring our story to a resolution that feels both satisfying and open-ended—a conclusion that answers questions while inviting us to ask new ones.</p>
    
    <p>This visual narrative reminds us that stories surround us every day, waiting to be discovered in the ordinary moments we might otherwise overlook. It invites us to look more closely at our world, to find the extraordinary in the everyday, and to create our own visual stories worth sharing.</p>
    """
    
    metadata = {
        "keywords": keywords,
        "hashtags": hashtags,
        "alt_texts": alt_texts,
        "meta_description": meta_description,
        "title": title
    }
    
    return blog_content, metadata

def generate_keywords(text: str, count: int = 7) -> List[str]:
    """Generate SEO keywords from text"""
    # Tokenize and clean the text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_freq = Counter(filtered_tokens)
    
    # Get the most common words
    common_words = [word for word, _ in word_freq.most_common(count)]
    
    # Add some bigrams for more specific keywords
    bigrams = []
    for i in range(len(filtered_tokens) - 1):
        if len(filtered_tokens[i]) > 3 and len(filtered_tokens[i+1]) > 3:
            bigrams.append(f"{filtered_tokens[i]} {filtered_tokens[i+1]}")
    
    bigram_freq = Counter(bigrams)
    for bigram, _ in bigram_freq.most_common(count - len(common_words)):
        if len(common_words) < count:
            common_words.append(bigram)
    
    # Ensure we have exactly the requested number of keywords
    if len(common_words) > count:
        common_words = common_words[:count]
    elif len(common_words) < count:
        # Add some engaging keywords if we don't have enough
        engaging_keywords = ["visual storytelling", "narrative photography", "visual journey", 
                           "image story", "visual narrative", "photography story", 
                           "visual perspective", "story in pictures"]
        for keyword in engaging_keywords:
            if keyword not in common_words and len(common_words) < count:
                common_words.append(keyword)
    
    return common_words

def generate_hashtags(keywords: List[str], count: int = 5) -> List[str]:
    """Generate hashtags from keywords"""
    hashtags = []
    
    # Convert keywords to hashtags
    for keyword in keywords:
        if len(hashtags) >= count:
            break
        
        # Remove spaces and special characters
        hashtag = re.sub(r'[^\w\s]', '', keyword)
        hashtag = hashtag.replace(' ', '')
        
        if hashtag and hashtag not in hashtags:
            hashtags.append(f"#{hashtag}")
    
    # Add some engaging hashtags if needed
    engaging_hashtags = ["#VisualStory", "#StorytellingArt", "#VisualNarrative", 
                        "#MomentsCaptured", "#VisualJourney", "#StoryInFrames", 
                        "#VisualExperience", "#ArtOfStory"]
    
    for hashtag in engaging_hashtags:
        if hashtag not in hashtags and len(hashtags) < count:
            hashtags.append(hashtag)
    
    return hashtags[:count]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)