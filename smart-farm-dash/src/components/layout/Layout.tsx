import React from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Navigation } from './Navigation';
import { useApp } from '@/contexts/AppContext';
import { cn } from '@/lib/utils';

export const Layout: React.FC = () => {
  const { mode } = useApp();

  return (
    <div className={cn(
      "min-h-screen bg-background",
      mode === 'farmer' ? "text-lg" : "text-base"
    )}>
      <Header />
      <Navigation />
      <main className={cn(
        "container mx-auto",
        mode === 'farmer' ? "px-4 py-6" : "px-6 py-8"
      )}>
        <Outlet />
      </main>
    </div>
  );
};