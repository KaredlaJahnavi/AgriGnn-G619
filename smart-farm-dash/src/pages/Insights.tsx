import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  TrendingUp, 
  AlertTriangle, 
  Lightbulb, 
  Calendar,
  ArrowUpRight,
  ArrowDownRight,
  Target,
  Zap,
  Shield
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Area, AreaChart } from 'recharts';

// Mock data
const trendData = [
  { month: 'Jan', yield: 18.2, weather: 85, market: 92 },
  { month: 'Feb', yield: 19.1, weather: 78, market: 88 },
  { month: 'Mar', yield: 20.5, weather: 82, market: 94 },
  { month: 'Apr', yield: 22.1, weather: 88, market: 96 },
  { month: 'May', yield: 23.8, weather: 92, market: 89 },
  { month: 'Jun', yield: 25.2, weather: 95, market: 91 },
];

const Insights: React.FC = () => {
  const keyInsights = [
    {
      title: 'Seasonal Yield Optimization',
      description: 'Rice yields show 15% improvement when planted between March-April with optimal rainfall conditions.',
      impact: 'High',
      category: 'Yield Optimization',
      trend: 'up',
      value: '+15%',
      icon: TrendingUp,
      color: 'text-success'
    },
    {
      title: 'Weather Risk Alert',
      description: 'Predicted El Niño conditions may reduce yields by 8-12% in the next season. Early preparation recommended.',
      impact: 'High',
      category: 'Risk Management',
      trend: 'down',
      value: '-10%',
      icon: AlertTriangle,
      color: 'text-warning'
    },
    {
      title: 'Soil Health Correlation',
      description: 'Farms with NDVI above 0.7 show 22% higher yields. Focus on vegetation health monitoring.',
      impact: 'Medium',
      category: 'Soil Management',
      trend: 'up',
      value: '+22%',
      icon: Lightbulb,
      color: 'text-primary'
    },
    {
      title: 'Market Timing Strategy',
      description: 'Historical data suggests harvesting 2 weeks earlier can increase profit margins by 18%.',
      impact: 'Medium',
      category: 'Market Strategy',
      trend: 'up',
      value: '+18%',
      icon: Target,
      color: 'text-accent-strong'
    }
  ];

  const predictiveAlerts = [
    {
      type: 'Weather',
      message: 'Heavy rainfall expected in North region - increase drainage measures',
      severity: 'high',
      affectedFarms: 1250,
      timeframe: '3-5 days',
      action: 'Immediate',
      icon: Shield
    },
    {
      type: 'Pest',
      message: 'Brown planthopper activity increasing - monitor and treat if necessary',
      severity: 'medium',
      affectedFarms: 850,
      timeframe: '1-2 weeks',
      action: 'Monitor',
      icon: Zap
    },
    {
      type: 'Market',
      message: 'Rice prices trending upward - optimal selling window approaching',
      severity: 'low',
      affectedFarms: 2100,
      timeframe: '2-3 weeks',
      action: 'Plan',
      icon: TrendingUp
    }
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'bg-destructive/20 text-destructive border-destructive/30';
      case 'medium': return 'bg-warning/20 text-warning border-warning/30';
      case 'low': return 'bg-success/20 text-success border-success/30';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'High': return 'bg-destructive text-destructive-foreground';
      case 'Medium': return 'bg-warning text-warning-foreground';
      case 'Low': return 'bg-success text-success-foreground';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        className="flex items-center justify-between"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div>
          <h1 className="text-3xl font-bold text-foreground">Agricultural Insights</h1>
          <p className="text-muted-foreground">AI-powered recommendations and predictive analytics</p>
        </div>
        <Button variant="outline">
          <Calendar className="w-4 h-4 mr-2" />
          Last 30 Days
        </Button>
      </motion.div>

      {/* Trend Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="shadow-soft">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-primary" />
              <span>Agricultural Trend Analysis</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="yieldGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Area
                  type="monotone"
                  dataKey="yield"
                  stroke="#22c55e"
                  fillOpacity={1}
                  fill="url(#yieldGradient)"
                  strokeWidth={2}
                />
                <Line type="monotone" dataKey="weather" stroke="#3b82f6" strokeWidth={2} />
                <Line type="monotone" dataKey="market" stroke="#f59e0b" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </motion.div>

      {/* Key Insights */}
      <motion.div
        className="space-y-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className="text-2xl font-bold text-foreground">Key Insights</h2>
        <div className="grid md:grid-cols-2 gap-4">
          {keyInsights.map((insight, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <Card className="shadow-soft hover:shadow-strong transition-all duration-300">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className="text-xs">
                      {insight.category}
                    </Badge>
                    <Badge className={getImpactColor(insight.impact)}>
                      {insight.impact} Impact
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg bg-gradient-primary`}>
                      <insight.icon className="w-5 h-5 text-primary-foreground" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-foreground">{insight.title}</h3>
                      <div className="flex items-center space-x-2">
                        {insight.trend === 'up' ? (
                          <ArrowUpRight className="w-4 h-4 text-success" />
                        ) : (
                          <ArrowDownRight className="w-4 h-4 text-destructive" />
                        )}
                        <span className={`text-sm font-medium ${
                          insight.trend === 'up' ? 'text-success' : 'text-destructive'
                        }`}>
                          {insight.value}
                        </span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {insight.description}
                  </p>
                  <Button variant="outline" size="sm" className="w-full">
                    Learn More
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Predictive Alerts */}
      <motion.div
        className="space-y-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-2xl font-bold text-foreground">Predictive Alerts</h2>
        <div className="space-y-3">
          {predictiveAlerts.map((alert, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
            >
              <Card className={`border-2 ${getSeverityColor(alert.severity)}`}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg ${getSeverityColor(alert.severity)}`}>
                        <alert.icon className="w-5 h-5" />
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center space-x-2">
                          <h3 className="font-semibold text-foreground">{alert.type} Alert</h3>
                          <Badge className={getSeverityColor(alert.severity)} variant="outline">
                            {alert.severity.toUpperCase()}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{alert.message}</p>
                        <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                          <span>Affected: {alert.affectedFarms.toLocaleString()} farms</span>
                          <span>Timeline: {alert.timeframe}</span>
                          <span>Action: {alert.action}</span>
                        </div>
                      </div>
                    </div>
                    <Button size="sm" variant="outline">
                      View Details
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Action Items */}
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <Card className="p-6 bg-gradient-sky text-accent-foreground">
          <h2 className="text-xl font-bold mb-4">Ready to Act on These Insights?</h2>
          <p className="text-sm mb-6 opacity-90">
            Get detailed reports and actionable recommendations for your agricultural operations
          </p>
          <div className="flex justify-center space-x-4">
            <Button variant="outline" className="bg-white/10 border-white/20 text-white hover:bg-white/20">
              Generate Report
            </Button>
            <Button className="bg-white text-accent-strong hover:bg-white/90">
              Schedule Consultation
            </Button>
          </div>
        </Card>
      </motion.div>
    </div>
  );
};

export default Insights;