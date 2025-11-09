import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getResearcherData, ResearcherData } from '@/services/api';
import { 
  TrendingUp, 
  Leaf,
  MapPin,
  Droplets
} from 'lucide-react';
import { 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';

// Mock data
const regionData = [
  { name: 'northRegion', value: 35, color: '#22c55e' },
  { name: 'southRegion', value: 28, color: '#3b82f6' },
  { name: 'west', value: 15, color: '#ef4444' },
];

const Dashboard: React.FC = () => {
  const { language } = useApp();
  const t = useTranslations(language);
  
  const [data, setData] = useState<ResearcherData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await getResearcherData();
        setData(response);
      } catch (error) {
        console.error('Error fetching researcher data:', error);
        // Set default data on error
        setData({
          totalPredictions: 0,
          averageYield: 0,
          regionDistribution: [],
          correlationData: []
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const kpiCards = [
    {
      title: t.averageYield,
      value: data ? `${data.averageYield} ${t.tonsPerHectare}` : `0 ${t.tonsPerHectare}`,
      icon: Leaf,
      color: 'text-success'
    },
    {
      title: t.predictionsMade,
      value: data ? data.totalPredictions.toLocaleString() : '0',
      icon: TrendingUp,
      color: 'text-accent-strong'
    }
  ];

  // Translate region data with backend data
  const getRegionTranslation = (name: string) => {
    const regionMap: { [key: string]: string } = {
      'north': t.northIndia || 'North India',
      'south': t.southIndia || 'South India',
      'east': t.eastIndia || 'East India',
      'west': t.westIndia || 'West India'
    };
    return regionMap[name] || name;
  };

  const translatedRegionData = data && data.regionDistribution.length > 0
    ? data.regionDistribution.map(region => {
        // Add default colors for regions
        const colorMap: { [key: string]: string } = {
          'north': '#22c55e',
          'south': '#3b82f6',
          'east': '#f59e0b',
          'west': '#ef4444'
        };
        
        return {
          name: region.name,
          value: region.value,
          color: colorMap[region.name] || '#888888',
          displayName: getRegionTranslation(region.name)
        };
      })
    : regionData.map(region => ({
        ...region,
        displayName: t[region.name] || region.name
      }));

  const correlationData = data?.correlationData || [
    { rainfall: 120, yield: 5.5, temperature: 28 },
    { rainfall: 95, yield: 4.5, temperature: 32 },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-4">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="mx-auto w-12 h-12 border-4 border-primary border-t-transparent rounded-full"
          />
          <p className="text-muted-foreground">Loading dashboard data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        className="flex flex-col md:flex-row md:items-center justify-between space-y-4 md:space-y-0"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-foreground">{t.dashboard || 'Research Dashboard'}</h1>
          <p className="text-muted-foreground">{t.dashboardSubtitle}</p>
        </div>
      </motion.div>

      {/* KPI Cards */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {kpiCards.map((kpi, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + index * 0.05 }}
            whileHover={{ scale: 1.02 }}
          >
            <Card className="shadow-soft hover:shadow-strong transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">{kpi.title}</p>
                    <p className="text-2xl font-bold text-foreground">{kpi.value}</p>
                  </div>
                  <div className={`p-3 rounded-lg bg-gradient-primary ${kpi.color}`}>
                    <kpi.icon className="w-6 h-6 text-primary-foreground" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Regional Distribution */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <MapPin className="w-5 h-5 text-primary" />
                <span>{t.regionalYieldDistribution}</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={translatedRegionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    dataKey="value"
                    label={({ displayName, value }) => `${displayName}: ${value}%`}
                  >
                    {translatedRegionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: any) => `${value}%`}
                    labelFormatter={(label: any) => {
                      const region = translatedRegionData.find(r => r.displayName === label);
                      return region ? region.displayName : label;
                    }}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>

        {/* Weather Correlation */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="shadow-soft">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Droplets className="w-5 h-5 text-primary" />
                <span>{t.rainfallYieldCorrelation}</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={correlationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="rainfall" name={t.rainfallMm} />
                  <YAxis dataKey="yield" name={t.yieldTonsHectare} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter dataKey="yield" fill="#22c55e" />
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;