import { useNavigate } from 'react-router-dom';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import { Button } from '@/components/ui/button';
import { PublicHeader } from '@/components/layout/PublicHeader';
import heroImage from '@/assets/hero-agriculture.jpg';

const Welcome = () => {
  const navigate = useNavigate();
  const { language } = useApp();
  const t = useTranslations(language);

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center z-0"
        style={{ backgroundImage: `url(${heroImage})` }}
      >
        <div className="absolute inset-0 bg-black/50" />
      </div>

      <PublicHeader />

      {/* Content */}
      <div className="relative z-10 text-center px-4 max-w-3xl mx-auto">
        <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 animate-fade-in">
          {t.welcomeTitle}
        </h1>
        <p className="text-xl md:text-2xl text-white/90 mb-12 animate-fade-in leading-relaxed">
          {t.welcomeDescription}
        </p>
        <Button
          onClick={() => navigate('/mode-selection')}
          size="lg"
          className="text-lg px-8 py-6 animate-scale-in hover-scale"
        >
          {t.getStarted}
        </Button>
      </div>
    </div>
  );
};

export default Welcome;
