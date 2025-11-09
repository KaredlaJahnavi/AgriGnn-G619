import { useNavigate } from 'react-router-dom';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { PublicHeader } from '@/components/layout/PublicHeader';
import { User, FlaskConical, ArrowLeft } from 'lucide-react';
import heroImage from '@/assets/hero-agriculture.jpg';

const ModeSelection = () => {
  const navigate = useNavigate();
  const { setMode, language } = useApp();
  const t = useTranslations(language);

  const handleFarmerMode = () => {
    setMode('farmer');
    navigate('/home');
  };

  const handleResearcherMode = () => {
    setMode('researcher');
    navigate('/signin');
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden p-4">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center z-0"
        style={{ backgroundImage: `url(${heroImage})` }}
      >
        <div className="absolute inset-0 bg-black/60" />
      </div>

      <PublicHeader />

      {/* Back Button - Fixed Top Left */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => navigate('/')}
        className="fixed top-20 left-4 z-20 bg-background/80 backdrop-blur-sm"
      >
        <ArrowLeft className="h-5 w-5" />
      </Button>

      {/* Content */}
      <div className="relative z-10 w-full max-w-5xl">
        <h1 className="text-4xl md:text-5xl font-bold text-white text-center mb-12 animate-fade-in">
          {t.selectMode}
        </h1>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Farmer Mode Card */}
          <Card 
            className="cursor-pointer transition-all hover:scale-105 animate-fade-in bg-card/95 backdrop-blur-sm"
            onClick={handleFarmerMode}
          >
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center">
                <User className="w-10 h-10 text-primary" />
              </div>
              <CardTitle className="text-2xl">{t.farmerModeTitle}</CardTitle>
              <CardDescription className="text-lg">
                {t.farmerModeDesc}
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground">
                {t.farmerMode}
              </p>
            </CardContent>
          </Card>

          {/* Researcher Mode Card */}
          <Card 
            className="cursor-pointer transition-all hover:scale-105 animate-fade-in bg-card/95 backdrop-blur-sm"
            onClick={handleResearcherMode}
          >
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center">
                <FlaskConical className="w-10 h-10 text-primary" />
              </div>
              <CardTitle className="text-2xl">{t.researcherModeTitle}</CardTitle>
              <CardDescription className="text-lg">
                {t.researcherModeDesc}
              </CardDescription>
            </CardHeader>
            <CardContent className="text-center">
              <p className="text-muted-foreground">
                {t.researcherMode}
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ModeSelection;