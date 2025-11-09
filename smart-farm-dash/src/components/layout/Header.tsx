import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Leaf, Globe, LogOut, User } from 'lucide-react';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import { useToast } from '@/hooks/use-toast';

export const Header: React.FC = () => {
  const { mode, setMode, language, setLanguage } = useApp();
  const t = useTranslations(language);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleLogout = () => {
    localStorage.removeItem('researcherUser');
    setMode('farmer');
    toast({
      title: t.logoutSuccess || 'Logged out successfully',
    });
    navigate('/signin');
  };

  const getUsername = () => {
    const userStr = localStorage.getItem('researcherUser');
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        return user.username || 'User';
      } catch {
        return 'User';
      }
    }
    return 'User';
  };

  return (
    <motion.header 
      className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Logo */}
        <motion.div 
          className="flex items-center space-x-2"
          whileHover={{ scale: 1.05 }}
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-primary">
            <Leaf className="w-6 h-6 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Agri-GNN</h1>
            <p className="text-xs text-muted-foreground">Smart Agriculture</p>
          </div>
        </motion.div>

        {/* Language Selector and Logout */}
        <div className="flex items-center space-x-4">
          {/* Language Selector */}
          <Select value={language} onValueChange={(value: string) => setLanguage(value as any)}>
            <SelectTrigger className="w-20">
              <Globe className="w-4 h-4 mr-1" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="en">EN</SelectItem>
              <SelectItem value="te">TE</SelectItem>
              <SelectItem value="hi">HI</SelectItem>
            </SelectContent>
          </Select>

          {/* User Dropdown (Researcher Mode Only) */}
          {mode === 'researcher' && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="flex items-center space-x-2"
                >
                  <User className="w-4 h-4" />
                  <span>{getUsername()}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onClick={handleLogout} className="flex items-center space-x-2 cursor-pointer">
                  <LogOut className="w-4 h-4" />
                  <span>{t.logout || 'Logout'}</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>
    </motion.header>
  );
};