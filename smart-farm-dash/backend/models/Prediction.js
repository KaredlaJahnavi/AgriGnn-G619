const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  // Farmer inputs
  soilType: { type: String, required: true },
  cropType: { type: String, required: true },
  rainfall: { type: Number, required: true },
  temperature: { type: Number, required: true },
  region: { type: String, required: true },
  irrigationUsed: { type: Boolean, required: true },
  fertilizerUsed: { type: Boolean, required: true },
  weatherCondition: { type: String, required: true },
  daysToHarvest: { type: Number, required: true },
  ndvi: { type: Number, required: true },
  evi: { type: Number, required: true },
  
  // Prediction result
  predictedYield: { type: Number, required: true },
  
  // Metadata
  timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Prediction', predictionSchema);


