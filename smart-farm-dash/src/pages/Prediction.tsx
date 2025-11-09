import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Download } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';

interface PredictionData {
  yieldAmount: number;
}

const Prediction: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { language } = useApp();
  const t = useTranslations(language);
  
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  
  const formData = location.state;

  useEffect(() => {
    // Check if predictedYield is already in formData (from backend)
    if (formData && formData.predictedYield) {
      setPrediction({
        yieldAmount: formData.predictedYield,
      });
    } else {
      // Fallback to mock prediction if no backend data
      setIsLoading(true);
      const timer = setTimeout(() => {
        const mockPrediction: PredictionData = {
          yieldAmount: generateYieldPrediction(formData),
        };
        setPrediction(mockPrediction);
        setIsLoading(false);
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [formData]);

  const generateYieldPrediction = (data: any) => {
    if (!data) return 3.77; // 15 q/acre ≈ 3.77 tons/hectare
    let baseYield = 3.77; // Base yield in tons/hectare
    
    // Soil type impact
    if (data.soilType === 'loamy') baseYield += 0.75;
    else if (data.soilType === 'clay') baseYield += 0.25;
    else baseYield -= 0.25;
    
    // Weather impact
    if (data.rainfall >= 75 && data.rainfall <= 150) baseYield += 0.50;
    else if (data.rainfall < 50 || data.rainfall > 180) baseYield -= 0.50;
    
    if (data.temperature >= 20 && data.temperature <= 30) baseYield += 0.25;
    else baseYield -= 0.25;
    
    // NDVI impact
    if (data.ndvi >= 0.7) baseYield += 0.75;
    else if (data.ndvi >= 0.4) baseYield += 0.25;
    else baseYield -= 0.50;
    
    // EVI impact
    if (data.evi >= 0.7) baseYield += 0.50;
    else if (data.evi >= 0.4) baseYield += 0.25;
    else baseYield -= 0.25;
    
    // Return value in tons/hectare (between 2.0 and 6.3)
    return Math.max(2.0, Math.min(6.3, baseYield + (Math.random() * 0.5 - 0.25)));
  };

  const getTranslatedValue = (key: string, value: string) => {
    // Map form values to translation keys
    const valueKey = value.toLowerCase().replace(/\s+/g, '');
    if (t[valueKey]) return t[valueKey];
    return value;
  };

  const downloadCSV = () => {
    if (!formData || !prediction) return;

    // Prepare CSV data with translated labels and values
    const csvData = [
      [t.region, getTranslatedValue('region', formData.region === 'north' ? t.northIndia : 
                                                formData.region === 'south' ? t.southIndia :
                                                formData.region === 'east' ? t.eastIndia : t.westIndia)],
      [t.irrigation, getTranslatedValue('irrigation', formData.irrigationMethod === 'true' ? t.yes : t.no)],
      [t.soilType, getTranslatedValue('soilType', 
        formData.soilType === 'clay' ? t.claysoil :
        formData.soilType === 'sandy' ? t.sandySoil :
        formData.soilType === 'loamy' ? t.loamySoil :
        formData.soilType === 'silt' ? t.siltSoil :
        formData.soilType === 'peaty' ? t.peatySoil : t.chalkySoil)],
      [t.cropType, getTranslatedValue('cropType',
        formData.cropType === 'rice' ? t.rice :
        formData.cropType === 'wheat' ? t.wheat :
        formData.cropType === 'maize' ? t.maize :
        formData.cropType === 'cotton' ? t.cotton :
        formData.cropType === 'soybean' ? t.soybean : t.barley)],
      [t.fertilizer, getTranslatedValue('fertilizer', formData.fertilizer === 'true' ? t.yes : t.no)],
      [t.weather, getTranslatedValue('weather',
        formData.weather === 'sunny' ? t.sunny :
        formData.weather === 'cloudy' ? t.cloudy : t.rainy)],
      [t.rainfall, `${formData.rainfall} mm`],
      [t.temperature, `${formData.temperature}°C`],
      [t.daysToHarvest, formData.daysToharvest],
      [t.vegetation, formData.ndvi],
      [t.evi, formData.evi],
      ['', ''],
      [t.predictedYield, `${prediction.yieldAmount.toFixed(1)} ${t.tonsPerHectare}`],
    ];

    // Convert to CSV string
    const csvContent = csvData.map(row => row.join(',')).join('\n');
    
    // Create blob and download
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `yield_prediction_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto">
        <motion.div
          className="text-center space-y-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Card className="p-8">
            <div className="space-y-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="mx-auto w-16 h-16 border-4 border-primary border-t-transparent rounded-full"
              />
              <h2 className="text-2xl font-bold">{t.predictionWizard}</h2>
              <Progress value={66} className="w-full max-w-md mx-auto" />
            </div>
          </Card>
        </motion.div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-4">{t.predictionWizard}</h2>
        <Button onClick={() => navigate('/input')}>
          {t.previous}
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Main Prediction Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="p-8 shadow-strong">
          <CardContent className="text-center space-y-8">
            <div className="space-y-4">
              <h1 className="text-2xl font-semibold text-muted-foreground">{t.predictedYield}</h1>
              <div className="flex items-center justify-center space-x-3">
                <span className="text-7xl font-bold text-primary">
                 {prediction.yieldAmount.toFixed(1)}
                </span>
                <div className="text-left">
                  <div className="text-xl font-medium text-foreground">
                    {t.tonsPerHectare}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Download Button */}
            <Button 
              onClick={downloadCSV}
              size="lg"
              className="shadow-soft"
            >
              <Download className="w-5 h-5 mr-2" />
              {t.downloadReport}
            </Button>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default Prediction;
