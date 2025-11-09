import React from 'react';
import { motion } from 'framer-motion';
import { NavLink, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { 
  Home, 
  Target, 
  BarChart3, 
  Lightbulb, 
  Smartphone,
  Monitor
} from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import { cn } from '@/lib/utils';

export const Navigation: React.FC = () => {
  const { mode, language } = useApp();
  const t = useTranslations(language);
  const location = useLocation();

  const farmerRoutes = [
    { path: '/home', icon: Home, label: t.home },
    { path: '/input', icon: Target, label: t.prediction },
  ];

  const researcherRoutes = [
    { path: '/home', icon: Home, label: t.home },
    { path: '/dashboard', icon: BarChart3, label: t.dashboard },
  ];

  const routes = mode === 'farmer' ? farmerRoutes : researcherRoutes;

  const isActive = (path: string) => {
    if (path === '/home') {
      return location.pathname === '/home';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <motion.nav 
      className="border-b bg-card"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="container px-4">
        <div className={cn(
          "flex items-center justify-center space-x-1",
          mode === 'farmer' ? "py-2" : "py-3"
        )}>
          {/* Mobile indicator */}
          <div className={cn(
            "flex items-center justify-center mr-2 p-1 rounded-md",
            mode === 'farmer' ? "bg-primary/10 md:hidden" : "hidden"
          )}>
            <Smartphone className="w-4 h-4 text-primary" />
          </div>

          {/* Desktop indicator */}
          <div className={cn(
            "items-center justify-center mr-2 p-1 rounded-md",
            mode === 'researcher' ? "hidden md:flex bg-accent/50" : "hidden"
          )}>
            <Monitor className="w-4 h-4 text-accent-strong" />
          </div>

          {routes.map((route, index) => {
            const active = isActive(route.path);
            return (
              <motion.div
                key={route.path}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.2, delay: index * 0.1 }}
              >
                <Button
                  asChild
                  variant={active ? "default" : "ghost"}
                  size={mode === 'farmer' ? "sm" : "default"}
                  className={cn(
                    "relative transition-all duration-300",
                    active && "shadow-soft",
                    mode === 'farmer' ? "text-xs px-2 py-1" : "text-sm px-3 py-2"
                  )}
                >
                  <NavLink to={route.path}>
                    <route.icon className={cn(
                      mode === 'farmer' ? "w-4 h-4" : "w-4 h-4 mr-2"
                    )} />
                    {mode === 'researcher' && <span>{route.label}</span>}
                    {mode === 'farmer' && (
                      <span className="ml-1 hidden sm:inline">{route.label}</span>
                    )}
                    
                    {active && (
                      <motion.div
                        className="absolute -bottom-1 left-1/2 w-6 h-0.5 bg-primary rounded-full"
                        layoutId="activeTab"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        style={{ transform: 'translateX(-50%)' }}
                      />
                    )}
                  </NavLink>
                </Button>
              </motion.div>
            );
          })}
        </div>
      </div>
    </motion.nav>
  );
};