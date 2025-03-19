# Trip Planner & Blog Creator

## Overview
Trip Planner & Blog Creator is a web application that helps users generate AI-powered travel itineraries and create travel blogs based on their images. The system integrates API calls to AI services (Gemini) for itinerary generation and blog creation. Users can view their plans on an interactive map and download them as a PDF.

## Features
- **AI-Powered Trip Itinerary**: Generates a complete travel plan based on user input.
- **Blog Creator**: Creates a storytelling travel blog based on images provided by user using BLIP model.
- **Interactive Map**: Displays travel locations dynamically.
- **PDF Export**: Allows users to download their travel itinerary or blog.

## Technologies Used
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask/Django/FastAPI)
- **AI Integration**: Gemini API, MAP API
- **Mapping**: Google Maps API
- **PDF Generation**: jsPDF

## Installation & Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/vasantha-kumar-s/trip-planner-ai.git
   cd trip-planner-ai
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up API Keys:**
   - Obtain API keys for Gemini and Google Map API
   - Set them in a `.env` file:
     ```sh
     GEMINI_API_KEY=your_gemini_api_key
     MAP_API=your_map_api_key
     ```
4. **Run the application:**
   ```sh
   python app.py  # or appropriate command for your framework
   ```
5. **Access the Web App:**
   Open `http://localhost:5000` in your browser.

## Usage
1. **Enter trip details**: Provide location, dates, budget and language.
2. **Generate itinerary**: The AI generates a complete travel plan.
3. **View map**: The locations are plotted on an interactive map.
4. **Download PDF**: Save your itinerary or blog for offline use.

## Future Enhancements
- **User Authentication**: Allow users to save their itineraries.
- **Customization Options**: More control over itinerary and blog styling.

## Contributions
Contributions are welcome! Feel free to submit pull requests or report issues.

## Contact
For any queries, reach out to vasanthakumarselvaraj04@gmail.com.

