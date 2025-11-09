import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface FarmerPredictionData {
  soilType: string;
  cropType: string;
  rainfall: number;
  temperature: number;
  region: string;
  irrigationUsed: string | boolean;
  fertilizerUsed: string | boolean;
  weatherCondition: string;
  daysToHarvest: number;
  ndvi: number;
  evi: number;
}

export interface PredictionResponse {
  predictedYield: number;
  id: string;
}

export interface ResearcherData {
  totalPredictions: number;
  averageYield: number;
  regionDistribution: Array<{ name: string; value: number }>;
  correlationData: Array<{ rainfall: number; yield: number; temperature: number }>;
}

// Farmer API
export const submitPrediction = async (data: FarmerPredictionData): Promise<PredictionResponse> => {
  const response = await api.post<PredictionResponse>('/api/farmer/predict', data);
  return response.data;
};

// Researcher API
export const getResearcherData = async (): Promise<ResearcherData> => {
  const response = await api.get<ResearcherData>('/api/researcher/data');
  return response.data;
};

export default api;


