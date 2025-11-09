import React from 'react';
import { PublicHeader } from '@/components/layout/PublicHeader';
import { useApp } from '@/contexts/AppContext';
import { useTranslations } from '@/utils/translations';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import heroImage from '@/assets/hero-agriculture.jpg';

const Contact = () => {
  const { language } = useApp();
  const t = useTranslations(language);

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

      {/* Content */}
      <Card className="w-full max-w-md relative z-10 bg-card/95 backdrop-blur-sm animate-fade-in mt-16">
        <CardHeader>
          <CardTitle className="text-2xl text-center">{t.contactUs}</CardTitle>
          <CardDescription className="text-center">{t.contactDescription}</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
            <div className="space-y-2">
              <Label htmlFor="name">{t.name}</Label>
              <Input
                id="name"
                type="text"
                placeholder={t.namePlaceholder}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">{t.email}</Label>
              <Input
                id="email"
                type="email"
                placeholder={t.emailPlaceholder}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="phone">{t.phoneNumber}</Label>
              <Input
                id="phone"
                type="tel"
                placeholder={t.phonePlaceholder}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="message">{t.message}</Label>
              <Textarea
                id="message"
                placeholder={t.messagePlaceholder}
                className="min-h-[100px]"
              />
            </div>

            <Button type="submit" className="w-full">
              {t.sendMessage}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default Contact;
