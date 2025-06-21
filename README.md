# Baybayin Chatbot & Translator Backend

A Flask-based backend for a Baybayin chatbot and translator using PyTorch Transformer models.

## Features

- **Translation**: Convert Filipino text to Baybayin script
- **Chatbot**: Interactive chatbot in Filipino/Baybayin
- **Text-to-Speech**: Audio output for translations
- **Multi-language Support**: English to Filipino translation

## API Endpoints

- `POST /translate` - Translate text to Baybayin
- `POST /chatbot` - Get chatbot response
- `GET /audio/<filename>` - Serve audio files

## Deployment on Render

### Prerequisites
- GitHub repository with your code
- Render account

### Steps

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create a new Web Service on Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

3. **Configure the service**
   - **Name**: `baybayin-chatbot-backend` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free

4. **Environment Variables** (optional)
   - No environment variables needed for basic functionality

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app

### Important Notes

- **Model Download**: The app automatically downloads models from Google Drive on first startup
- **Free Tier Limitations**: 
  - 750 hours/month
  - Service sleeps after 15 minutes of inactivity
  - First request after sleep may take 30-60 seconds
- **CORS**: Already configured for frontend integration

## Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server**
   ```bash
   python app.py
   ```

3. **Test endpoints**
   - Translation: `POST http://localhost:5050/translate`
   - Chatbot: `POST http://localhost:5050/chatbot`

## Model Information

- **Translation Model**: Transformer-based Filipino to Baybayin translator
- **Chatbot Model**: Transformer-based conversational AI
- **Tokenizers**: Character-level tokenizers for both models
- **Device**: Automatically uses GPU if available, falls back to CPU

## File Structure

```
backend-clean/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── runtime.txt           # Python version specification
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── Chatbot/             # Chatbot models (downloaded automatically)
└── Translation/         # Translation models (downloaded automatically)
``` 