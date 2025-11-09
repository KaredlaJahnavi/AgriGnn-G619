# Smart Farm Dashboard - Startup Guide

## Prerequisites
- Node.js and npm installed
- MongoDB connection string (MONGO_URI in backend/.env)

## Starting the Application

### Backend (Terminal 1)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the backend server:
   ```bash
   npm start
   ```

The backend will run on `http://localhost:5000`

### Frontend (Terminal 2)
1. Navigate to the project root:
   ```bash
   cd ..
   ```

2. Install dependencies (if not already installed):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will run on `http://localhost:5173` (or similar)

## Features

### Farmer Mode
- Fill in farm input fields (soil type, crop type, rainfall, temperature, region, irrigation, fertilizer, weather, days to harvest, NDVI, EVI)
- Click "Predict" to get yield prediction
- Prediction is stored in MongoDB
- Download prediction report as CSV

### Researcher Mode
- View real-time KPI cards showing:
  - Average yield across all predictions
  - Total number of predictions made
- View regional yield distribution pie chart
- View rainfall vs yield correlation scatter plot
- Dashboard updates automatically on refresh with latest data from MongoDB

## API Endpoints

### POST /api/farmer/predict
Submit farmer inputs and receive predicted yield.

### GET /api/researcher/data
Get aggregated statistics and charts data.

## Environment Variables

Create a `.env` file in the `backend` directory with:
```
MONGO_URI=your_mongodb_connection_string
PORT=5000
```



