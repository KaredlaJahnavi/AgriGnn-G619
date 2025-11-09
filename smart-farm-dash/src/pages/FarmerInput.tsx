import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { 
  ArrowLeft, 
  ArrowRight, 
  Mountain, 
  Waves, 
  TreePine, 
  Cloud, 
  Thermometer, 
  Leaf,
  CheckCircle
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import { submitPrediction, FarmerPredictionData } from '@/services/api';

interface FormData {
  soilType: string;
  rainfall: number;
  temperature: number;
  weather: string;
  daysToharvest: string;
  ndvi: number;
  evi: number;
  region: string;
  cropType: string;
  fertilizer: string;
  irrigationMethod: string;
}

const FarmerInput: React.FC = () => {
  const navigate = useNavigate();
  const { language } = useApp();
  const t = useTranslations(language);
  
  const [currentStep, setCurrentStep] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    soilType: '',
    rainfall: 500,
    temperature: 27,
    weather: '',
    daysToharvest: '120',
    ndvi: 0.45,
    evi: 0.27,
    region: '',
    cropType: '',
    fertilizer: '',
    irrigationMethod: '',
  });

  // Validation errors state
  const [errors, setErrors] = useState({
    rainfall: '',
    temperature: '',
    daysToharvest: '',
    ndvi: '',
    evi: '',
  });

  const steps = [
    {
      title: t.farmDetails,
      icon: Mountain,
      component: FarmDetailsStep,
    },
    {
      title: t.soilCrop,
      icon: Leaf,
      component: SoilCropStep,
    },
    {
      title: t.weather,
      icon: Cloud,
      component: WeatherStep,
    },
    {
      title: t.vegetation,
      icon: Leaf,
      component: VegetationStep,
    },
  ];

  const progress = ((currentStep + 1) / steps.length) * 100;

  const handleNext = async () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Submit prediction to backend
      setIsLoading(true);
      try {
        const predictionData: FarmerPredictionData = {
          soilType: formData.soilType,
          cropType: formData.cropType,
          rainfall: formData.rainfall,
          temperature: formData.temperature,
          region: formData.region,
          irrigationUsed: formData.irrigationMethod === 'true',
          fertilizerUsed: formData.fertilizer === 'true',
          weatherCondition: formData.weather,
          daysToHarvest: parseInt(formData.daysToharvest),
          ndvi: formData.ndvi,
          evi: formData.evi
        };
        
        const response = await submitPrediction(predictionData);
        
        // Navigate to prediction with form data and predicted yield
        navigate('/prediction', { 
          state: { 
            ...formData, 
            predictedYield: response.predictedYield 
          } 
        });
      } catch (error) {
        console.error('Prediction error:', error);
        alert('Failed to get prediction. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const isStepValid = () => {
    switch (currentStep) {
      case 0:
        return formData.region !== '' && formData.irrigationMethod !== '';
      case 1:
        return formData.soilType !== '' && formData.cropType !== '' && formData.fertilizer !== '';
      case 2:
        return formData.weather !== '' && 
               formData.rainfall >= 100 && formData.rainfall <= 1000 && 
               formData.temperature >= 15 && formData.temperature <= 40 && 
               formData.daysToharvest !== '' && 
               parseInt(formData.daysToharvest) >= 60 && parseInt(formData.daysToharvest) <= 149 &&
               errors.rainfall === '' && errors.temperature === '' && errors.daysToharvest === '';
      case 3:
        return formData.ndvi >= 0.38 && formData.ndvi <= 0.57 && 
               formData.evi >= 0.23 && formData.evi <= 0.34 && 
               errors.ndvi === '' && errors.evi === '';
      default:
        return false;
    }
  };

  const CurrentStepComponent = steps[currentStep].component;

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Progress Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-3xl font-bold text-foreground">{t.predictionWizard}</h1>
        <div className="space-y-2">
          <Progress value={progress} className="h-3" />
          <p className="text-sm text-muted-foreground">
            {t.step} {currentStep + 1} {t.of} {steps.length}
          </p>
        </div>
      </motion.div>

      {/* Step Indicator */}
      <div className="flex justify-center space-x-4">
        {steps.map((step, index) => {
          const StepIcon = step.icon;
          const isActive = index === currentStep;
          const isCompleted = index < currentStep;
          
          return (
            <motion.div
              key={index}
              className={`flex items-center space-x-2 p-2 rounded-lg ${
                isActive ? 'bg-primary text-primary-foreground' :
                isCompleted ? 'bg-success text-success-foreground' :
                'bg-muted text-muted-foreground'
              }`}
              whileHover={{ scale: 1.05 }}
            >
              {isCompleted ? (
                <CheckCircle className="w-5 h-5" />
              ) : (
                <StepIcon className="w-5 h-5" />
              )}
              <span className="text-sm font-medium hidden sm:inline">{step.title}</span>
            </motion.div>
          );
        })}
      </div>

      {/* Step Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="shadow-strong">
            <CardHeader className="text-center">
              <CardTitle className="text-2xl flex items-center justify-center space-x-2">
                {React.createElement(steps[currentStep].icon, { className: "w-6 h-6 text-primary" })}
                <span>{steps[currentStep].title}</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <CurrentStepComponent 
                formData={formData} 
                setFormData={setFormData}
                translations={t}
                errors={errors}
                setErrors={setErrors}
              />
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>

      {/* Navigation Buttons */}
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={handlePrevious}
          disabled={currentStep === 0}
          size="lg"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          {t.previous}
        </Button>
        
        <Button
          onClick={handleNext}
          disabled={!isStepValid() || isLoading}
          size="lg"
          className="shadow-soft"
        >
          {isLoading ? (
            <>Loading...</>
          ) : (
            <>
              {currentStep === steps.length - 1 ? t.getPrediction : t.next}
              <ArrowRight className="w-4 h-4 ml-2" />
            </>
          )}
        </Button>
      </div>
    </div>
  );
};

// Farm Details Step Component
const FarmDetailsStep: React.FC<{ 
  formData: FormData; 
  setFormData: (data: FormData) => void; 
  translations: any;
  errors?: any;
  setErrors?: any;
}> = ({
  formData,
  setFormData,
  translations: t
}) => {
  return (
    <div className="space-y-6">
      <p className="text-muted-foreground text-center mb-6">
        {t.farmDetailsDesc}
      </p>
      
      <div className="grid gap-6">
        {/* Region Selection */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.region}</Label>
          <Select value={formData.region} onValueChange={(value) => setFormData({ ...formData, region: value })}>
            <SelectTrigger className="h-12">
              <SelectValue placeholder={t.selectRegion} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="north">{t.northIndia}</SelectItem>
              <SelectItem value="south">{t.southIndia}</SelectItem>
              <SelectItem value="east">{t.eastIndia}</SelectItem>
              <SelectItem value="west">{t.westIndia}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Irrigation Method */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.irrigation}</Label>
          <Select value={formData.irrigationMethod} onValueChange={(value) => setFormData({ ...formData, irrigationMethod: value })}>
            <SelectTrigger className="h-12">
              <SelectValue placeholder={t.selectIrrigation} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="true">{t.yes}</SelectItem>
              <SelectItem value="false">{t.no}</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
};

// Soil & Crop Step Component
const SoilCropStep: React.FC<{ 
  formData: FormData; 
  setFormData: (data: FormData) => void; 
  translations: any;
  errors?: any;
  setErrors?: any;
}> = ({
  formData,
  setFormData,
  translations: t
}) => {
  return (
    <div className="space-y-6">
      <p className="text-muted-foreground text-center mb-6">
        {t.soilCropDesc}
      </p>
      
      <div className="grid gap-6">
        {/* Soil Type */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.soilType}</Label>
          <Select value={formData.soilType} onValueChange={(value) => setFormData({ ...formData, soilType: value })}>
            <SelectTrigger className="h-12">
              <SelectValue placeholder={t.selectSoilType} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="clay">{t.claysoil}</SelectItem>
              <SelectItem value="sandy">{t.sandySoil}</SelectItem>
              <SelectItem value="loamy">{t.loamySoil}</SelectItem>
              <SelectItem value="silt">{t.siltSoil}</SelectItem>
              <SelectItem value="peaty">{t.peatySoil}</SelectItem>
              <SelectItem value="chalky">{t.chalkySoil}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Crop Type */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.cropType}</Label>
          <Select value={formData.cropType} onValueChange={(value) => setFormData({ ...formData, cropType: value })}>
            <SelectTrigger className="h-12">
              <SelectValue placeholder={t.selectCropType} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="rice">{t.rice}</SelectItem>
              <SelectItem value="wheat">{t.wheat}</SelectItem>
              <SelectItem value="maize">{t.maize}</SelectItem>
              <SelectItem value="cotton">{t.cotton}</SelectItem>
              <SelectItem value="soybean">{t.soybean}</SelectItem>
              <SelectItem value="barley">{t.barley}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Fertilizer */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.fertilizer}</Label>
          <Select value={formData.fertilizer} onValueChange={(value) => setFormData({ ...formData, fertilizer: value })}>
            <SelectTrigger className="h-12">
              <SelectValue placeholder={t.selectFertilizer} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="true">{t.yes}</SelectItem>
              <SelectItem value="false">{t.no}</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
};

// Weather Step Component
const WeatherStep: React.FC<{ 
  formData: FormData; 
  setFormData: (data: FormData) => void; 
  translations: any;
  errors: any;
  setErrors: any;
}> = ({
  formData,
  setFormData,
  translations: t,
  errors,
  setErrors
}) => {
  const validateRainfall = (value: number) => {
    if (isNaN(value) || value < 100 || value > 1000) {
      setErrors({ ...errors, rainfall: t.rainfallValidation });
    } else {
      setErrors({ ...errors, rainfall: '' });
    }
  };

  const validateTemperature = (value: number) => {
    if (isNaN(value) || value < 15 || value > 40) {
      setErrors({ ...errors, temperature: t.temperatureValidation });
    } else {
      setErrors({ ...errors, temperature: '' });
    }
  };

  const validateDaysToHarvest = (value: string) => {
    const numValue = parseInt(value);
    if (!value || isNaN(numValue) || numValue < 60 || numValue > 149) {
      setErrors({ ...errors, daysToharvest: t.daysToHarvestValidation });
    } else {
      setErrors({ ...errors, daysToharvest: '' });
    }
  };

  return (
    <div className="space-y-8">
      <p className="text-muted-foreground text-center mb-6">
        {t.weatherDesc}
      </p>

      <div className="space-y-6">
        {/* Weather Condition */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.weather}</Label>
          <Select value={formData.weather} onValueChange={(value) => setFormData({ ...formData, weather: value })}>
            <SelectTrigger className="h-12">
              <SelectValue placeholder={t.selectWeather} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="sunny">{t.sunny}</SelectItem>
              <SelectItem value="cloudy">{t.cloudy}</SelectItem>
              <SelectItem value="rainy">{t.rainy}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Rainfall */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Cloud className="w-5 h-5 text-primary" />
            <Label className="text-lg font-medium">{t.rainfall || 'Rainfall (mm)'}</Label>
          </div>
          <div className="space-y-2">
            <Input
              type="number"
              min="100"
              max="1000"
              step="10"
              value={formData.rainfall}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setFormData({ ...formData, rainfall: val });
                validateRainfall(val);
              }}
              placeholder={t.enterRainfallPlaceholder || 'Enter Rainfall (mm)'}
              className="h-12 text-lg [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
            />
            {errors.rainfall && <p className="text-xs text-red-500">{errors.rainfall}</p>}
          </div>
        </div>

        {/* Temperature */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Thermometer className="w-5 h-5 text-primary" />
            <Label className="text-lg font-medium">{t.temperature || 'Temperature (°C)'}</Label>
          </div>
          <div className="space-y-2">
            <Input
              type="number"
              min="15"
              max="40"
              step="1"
              value={formData.temperature}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setFormData({ ...formData, temperature: val });
                validateTemperature(val);
              }}
              placeholder={t.enterTemperaturePlaceholder || 'Enter Temperature (°C)'}
              className="h-12 text-lg [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
            />
            {errors.temperature && <p className="text-xs text-red-500">{errors.temperature}</p>}
          </div>
        </div>

        {/* Days to Harvest */}
        <div className="space-y-2">
          <Label className="text-lg font-medium">{t.daysToHarvest}</Label>
          <Input
            type="number"
            min="60"
            max="149"
            value={formData.daysToharvest}
            onChange={(e) => {
              const val = e.target.value;
              setFormData({ ...formData, daysToharvest: val });
              validateDaysToHarvest(val);
            }}
            placeholder={t.enterDaysToHarvest}
            className="h-12 text-lg [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none [-moz-appearance:textfield]"
          />
          {errors.daysToharvest && <p className="text-xs text-red-500">{errors.daysToharvest}</p>}
        </div>
      </div>
    </div>
  );
};

// Vegetation Step Component
const VegetationStep: React.FC<{ 
  formData: FormData; 
  setFormData: (data: FormData) => void; 
  translations: any;
  errors: any;
  setErrors: any;
}> = ({
  formData,
  setFormData,
  translations: t,
  errors,
  setErrors
}) => {
  const validateNDVI = (value: number) => {
    if (isNaN(value) || value < 0.38 || value > 0.57) {
      setErrors({ ...errors, ndvi: t.ndviValidation });
    } else {
      setErrors({ ...errors, ndvi: '' });
    }
  };

  const validateEVI = (value: number) => {
    if (isNaN(value) || value < 0.23 || value > 0.34) {
      setErrors({ ...errors, evi: t.eviValidation });
    } else {
      setErrors({ ...errors, evi: '' });
    }
  };

  return (
    <div className="space-y-6">
      <p className="text-muted-foreground text-center mb-6">
        {t.vegetationDesc}
      </p>

      <div className="space-y-6">
        {/* Manual NDVI Input */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Leaf className="w-5 h-5 text-primary" />
            <Label className="text-lg font-medium">{t.ndviValue}</Label>
          </div>
          <div className="space-y-2">
            <Input
              type="number"
              min="0.38"
              max="0.57"
              step="0.01"
              value={formData.ndvi}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setFormData({ ...formData, ndvi: val });
                validateNDVI(val);
              }}
              placeholder={t.enterVegetation}
              className="text-lg p-4"
            />
            {errors.ndvi && <p className="text-xs text-red-500">{errors.ndvi}</p>}
            {!errors.ndvi && <p className="text-sm text-muted-foreground">{t.ndviDesc}</p>}
          </div>
        </div>

        {/* Manual EVI Input */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2">
            <Leaf className="w-5 h-5 text-primary" />
            <Label className="text-lg font-medium">{t.eviValue}</Label>
          </div>
          <div className="space-y-2">
            <Input
              type="number"
              min="0.23"
              max="0.34"
              step="0.01"
              value={formData.evi}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setFormData({ ...formData, evi: val });
                validateEVI(val);
              }}
              placeholder={t.enterEvi}
              className="text-lg p-4"
            />
            {errors.evi && <p className="text-xs text-red-500">{errors.evi}</p>}
            {!errors.evi && <p className="text-sm text-muted-foreground">{t.eviDesc}</p>}
          </div>
        </div>

      </div>
    </div>
  );
};

export default FarmerInput;