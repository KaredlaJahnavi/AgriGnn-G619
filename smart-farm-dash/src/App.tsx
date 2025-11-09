import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppProvider } from "@/contexts/AppContext";
import { Layout } from "@/components/layout/Layout";
import Welcome from "./pages/Welcome";
import ModeSelection from "./pages/ModeSelection";
import SignUp from "./pages/SignUp";
import SignIn from "./pages/SignIn";
import Contact from "./pages/Contact";
import Home from "./pages/Home";
import FarmerInput from "./pages/FarmerInput";
import Prediction from "./pages/Prediction";
import Dashboard from "./pages/Dashboard";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AppProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            {/* Public routes without Layout */}
            <Route path="/" element={<Welcome />} />
            <Route path="/mode-selection" element={<ModeSelection />} />
            <Route path="/signup" element={<SignUp />} />
            <Route path="/signin" element={<SignIn />} />
            <Route path="/contact" element={<Contact />} />
            
            {/* App routes with Layout */}
            <Route path="/" element={<Layout />}>
              <Route path="/home" element={<Home />} />
              <Route path="/input" element={<FarmerInput />} />
              <Route path="/prediction" element={<Prediction />} />
              <Route path="/dashboard" element={<Dashboard />} />
            </Route>
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </AppProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
