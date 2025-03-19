// server.js
const express = require('express');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Google Gemini API setup
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Helper function to extract JSON from text that might include markdown code blocks
function extractJsonFromText(text) {
    // Try to match JSON inside markdown code blocks (```json ... ```)
    const markdownJsonMatch = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (markdownJsonMatch && markdownJsonMatch[1]) {
        return markdownJsonMatch[1];
    }
    
    // Try to match standalone JSON object
    const jsonMatch = text.match(/(\{[\s\S]*\})/);
    if (jsonMatch && jsonMatch[1]) {
        return jsonMatch[1];
    }
    
    return text; // Return original text if no matches
}

// API endpoint to generate trip plan
app.post('/api/generate-trip', async (req, res) => {
    try {
        const { destination, language, duration, transport, hotelBudget, foodBudget, people } = req.body;
        
        // Create prompt for Gemini API with separate hotel and food budgets
        const prompt = `Create a detailed trip plan for ${destination} for ${duration} days. 
        The plan should use ${transport} as the primary mode of transportation for ${people}.
        The daily hotel budget is ${hotelBudget} per night and the food budget is ${foodBudget} per meal.
        Please provide the response in ${language} language.
        
        IMPORTANT: I need the response ONLY as a raw JSON object with no markdown formatting, code blocks, or explanations.
        
        The JSON structure should be as follows:
        {
          "schedule": [
            {
              "description": "Brief overview of the day",
              "activities": ["Activity 1", "Activity 2", "Activity 3"],
              "restaurants": ["Restaurant 1 (approximate price range)", "Restaurant 2 (approximate price range)"]
            }
          ],
          "hotels": {
            "cheap": ["Budget Hotel 1 (price per night)", "Budget Hotel 2 (price per night)", "Budget Hotel 3 (price per night)"],
            "affordable": ["Mid-range Hotel 1 (price per night)", "Mid-range Hotel 2 (price per night)", "Mid-range Hotel 3 (price per night)"],
            "expensive": ["Luxury Hotel 1 (price per night)", "Luxury Hotel 2 (price per night)", "Luxury Hotel 3 (price per night)"]
          },
          "localFood": ["Local dish 1 (approximate price)", "Local dish 2 (approximate price)", "Local dish 3 (approximate price)", "Local dish 4 (approximate price)", "Local dish 5 (approximate price)"],
          "budgetAnalysis": {
            "hotelOptions": "Brief analysis of hotel options that match the ${hotelBudget} per night budget",
            "foodOptions": "Brief analysis of food options that match the ${foodBudget} per meal budget",
            "recommendedOptions": "Specific recommendations within budget"
          }
        }
        
        The schedule array should have exactly ${duration} days.
        For each day, provide 3-5 activities and 2-3 restaurant recommendations with approximate price ranges.
        For hotels, provide exactly 3 options for each price category with approximate prices per night.
        Analyze if the ${hotelBudget} per night budget is low, medium, or high for this destination and recommend appropriate options.
        Analyze if the ${foodBudget} per meal budget is low, medium, or high for this destination and recommend appropriate options.
        For local food, provide 5 must-try local dishes or specialties with approximate prices.
        
        Return ONLY the raw JSON with no extra text or formatting.`;

        // Call Gemini API
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();
        
        console.log("Raw response from Gemini:", text);
        
        // Extract and parse the JSON response
        try {
            // Try to extract JSON from text that might include markdown code blocks
            const cleanedJsonText = extractJsonFromText(text);
            console.log("Cleaned JSON text:", cleanedJsonText);
            
            const jsonResponse = JSON.parse(cleanedJsonText);
            res.json(jsonResponse);
        } catch (jsonError) {
            console.error('Error parsing JSON response:', jsonError);
            res.status(500).json({ 
                error: 'Failed to parse JSON from Gemini response', 
                details: jsonError.message,
                rawResponse: text
            });
        }
    } catch (error) {
        console.error('Error generating trip plan:', error);
        res.status(500).json({ error: 'Failed to generate trip plan', details: error.message });
    }
});

// Serve main.html as the default page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'main.html'));
});

// Add explicit route for index.html
app.get('/index.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Handle all other routes by serving the appropriate HTML file
app.get('*', (req, res) => {
    // Check if the requested file exists in the public directory
    const filePath = path.join(__dirname, 'public', req.path);
    res.sendFile(filePath, (err) => {
        if (err) {
            // If file not found, serve main.html
            res.sendFile(path.join(__dirname, 'public', 'main.html'));
        }
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}/main.html`);
});