const express = require('express');
const router = express.Router();
const Prediction = require('../models/Prediction');

// GET /api/researcher/data
router.get('/data', async (req, res) => {
  try {
    // Get all predictions
    const predictions = await Prediction.find({});

    // Calculate total number of predictions
    const totalPredictions = predictions.length;

    // Calculate average yield
    let averageYield = 0;
    if (predictions.length > 0) {
      const totalYield = predictions.reduce((sum, p) => sum + p.predictedYield, 0);
      averageYield = totalYield / predictions.length;
    }

    // Regional yield distribution for pie chart
    const regionDistribution = {};
    predictions.forEach(prediction => {
      if (!regionDistribution[prediction.region]) {
        regionDistribution[prediction.region] = 0;
      }
      regionDistribution[prediction.region] += prediction.predictedYield;
    });

    // Convert to array format for pie chart
    // Calculate percentages based on total yield per region
    const totalYieldSum = Object.values(regionDistribution).reduce((sum, val) => sum + val, 0);
    const regionData = Object.entries(regionDistribution).map(([region, yieldSum]) => ({
      name: region,
      value: totalYieldSum > 0 ? parseFloat(((yieldSum / totalYieldSum) * 100).toFixed(2)) : 0
    }));

    // Rainfall vs yield correlation data for scatter plot
    const correlationData = predictions.map(p => ({
      rainfall: p.rainfall,
      yield: parseFloat(p.predictedYield.toFixed(2)),
      temperature: p.temperature
    }));

    // Return aggregated data
    res.json({
      totalPredictions,
      averageYield: parseFloat(averageYield.toFixed(2)),
      regionDistribution: regionData,
      correlationData
    });
  } catch (error) {
    console.error('Researcher data error:', error);
    res.status(500).json({ error: 'Internal server error', message: error.message });
  }
});

module.exports = router;


