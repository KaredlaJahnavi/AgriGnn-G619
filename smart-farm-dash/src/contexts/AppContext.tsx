import React, { createContext, useContext, useState, ReactNode } from 'react';

export type AppMode = 'farmer' | 'researcher';
export type Language = 'en' | 'te' | 'hi';

interface AppContextType {
  mode: AppMode;
  setMode: (mode: AppMode) => void;
  language: Language;
  setLanguage: (language: Language) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const useApp = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [mode, setMode] = useState<AppMode>('farmer');
  const [language, setLanguage] = useState<Language>('en');

  const value = {
    mode,
    setMode,
    language,
    setLanguage,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};