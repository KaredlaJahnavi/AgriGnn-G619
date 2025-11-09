import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Globe } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';

export const LanguageSelector: React.FC = () => {
  const { language, setLanguage } = useApp();

  return (
    <Select value={language} onValueChange={(value: string) => setLanguage(value as any)}>
      <SelectTrigger className="w-24 bg-background/80 backdrop-blur-sm">
        <Globe className="w-4 h-4 mr-1" />
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="en">EN</SelectItem>
        <SelectItem value="te">TE</SelectItem>
        <SelectItem value="hi">HI</SelectItem>
      </SelectContent>
    </Select>
  );
};
