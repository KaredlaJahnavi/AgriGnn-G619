import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { ArrowRight, Leaf, TrendingUp, Shield, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import heroImage from '@/assets/hero-agriculture.jpg';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const { mode, language } = useApp();
  const t = useTranslations(language);
  const [isLearnMoreOpen, setIsLearnMoreOpen] = useState(false);

  const features = [
    {
      icon: TrendingUp,
      title: mode === 'farmer' ? t.smartPredictions : t.advancedAnalytics,
      description: mode === 'farmer' 
        ? t.smartPredictionsDesc
        : t.advancedAnalyticsDesc,
      color: 'text-success'
    },
    {
      icon: Shield,
      title: mode === 'farmer' ? t.riskManagement : t.riskAssessment,
      description: mode === 'farmer'
        ? t.riskManagementDesc
        : t.riskAssessmentDesc,
      color: 'text-warning'
    },
    {
      icon: Zap,
      title: mode === 'farmer' ? t.quickResults : t.realTimeInsights,
      description: mode === 'farmer'
        ? t.quickResultsDesc
        : t.realTimeInsightsDesc,
      color: 'text-accent-strong'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.section
        className="relative overflow-hidden rounded-2xl"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        {/* Background Image */}
        <div 
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroImage})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-primary/90 via-primary/70 to-primary/50" />
        </div>

        {/* Content */}
        <div className="relative z-10 px-6 py-16 md:px-12 md:py-24">
          <div className="max-w-4xl">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <h1 className={`font-bold text-primary-foreground mb-6 ${
                mode === 'farmer' ? 'text-3xl md:text-5xl' : 'text-4xl md:text-6xl'
              }`}>
                {t.heroTitle}
              </h1>
              <p className={`text-primary-foreground/90 mb-8 leading-relaxed ${
                mode === 'farmer' ? 'text-lg md:text-xl' : 'text-xl md:text-2xl'
              }`}>
                {t.heroSubtitle}
              </p>
            </motion.div>

            <motion.div
              className="flex flex-col sm:flex-row gap-4"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              {mode === 'farmer' && (
                <Button
                  size="lg"
                  className="bg-white text-primary hover:bg-white/90 shadow-strong"
                  onClick={() => navigate('/input')}
                >
                  <Leaf className="w-5 h-5 mr-2" />
                  {t.startPrediction}
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              )}
              
              <Button
                variant="outline"
                size={mode === 'farmer' ? 'lg' : 'default'}
                className="bg-transparent border-white text-white hover:bg-white/10"
                onClick={() => setIsLearnMoreOpen(true)}
              >
                {t.learnMore}
              </Button>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Features Section */}
      <motion.section
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.6 }}
      >
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <Card className="h-full border-0 shadow-soft hover:shadow-strong transition-all duration-300">
                <CardContent className="p-6">
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-primary mb-4`}>
                    <feature.icon className="w-6 h-6 text-primary-foreground" />
                  </div>
                  <h3 className="text-xl font-semibold mb-3 text-foreground">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Quick Actions for Farmer Mode */}
      {mode === 'farmer' && (
        <motion.section
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.0 }}
        >
          <Card className="p-8 bg-gradient-earth text-secondary-foreground">
            <h2 className="text-2xl font-bold mb-4">{t.readyToStart}</h2>
            <p className="text-lg mb-6 opacity-90">
              {t.getYieldInSteps}
            </p>
            <Button 
              size="lg"
              className="bg-white text-secondary-rich hover:bg-white/90"
              onClick={() => navigate('/input')}
            >
              {t.startNow}
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </Card>
        </motion.section>
      )}

      {/* Learn More Modal */}
      <Dialog open={isLearnMoreOpen} onOpenChange={setIsLearnMoreOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-primary">
              {t.aboutPlatform}
            </DialogTitle>
            <DialogDescription className="text-lg leading-relaxed mt-4">
              {t.aboutPlatformDesc}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end mt-6">
            <Button onClick={() => setIsLearnMoreOpen(false)}>
              {t.gotIt}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Home;