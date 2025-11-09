import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { LanguageSelector } from '@/components/LanguageSelector';
import { Leaf, Home, Menu, Mail } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';

export const PublicHeader: React.FC = () => {
  const navigate = useNavigate();
  const { language } = useApp();
  const t = useTranslations(language);

  return (
    <motion.header 
      className="fixed top-0 left-0 right-0 z-50 w-full bg-background/80 backdrop-blur-sm border-b"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Logo */}
        <motion.div 
          className="flex items-center space-x-2 cursor-pointer"
          whileHover={{ scale: 1.05 }}
          onClick={() => navigate('/')}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-primary">
            <Leaf className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Agri-GNN</h1>
            <p className="text-xs text-muted-foreground">Smart Agriculture</p>
          </div>
        </motion.div>

        {/* Navigation and Language Selector */}
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/')}
            className="hidden sm:flex items-center space-x-1"
          >
            <Home className="w-4 h-4" />
            <span>{t.home}</span>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/mode-selection')}
            className="hidden sm:flex items-center space-x-1"
          >
            <Menu className="w-4 h-4" />
            <span>{t.selectMode}</span>
          </Button>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/contact')}
            className="hidden sm:flex items-center space-x-1"
          >
            <Mail className="w-4 h-4" />
            <span>{t.contact}</span>
          </Button>

          <LanguageSelector />
        </div>
      </div>
    </motion.header>
  );
};