const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const dns = require('dns');

dotenv.config({ path: require('path').join(__dirname, '.env') });

const authRoutes = require('./routes/auth');
const farmerRoutes = require('./routes/farmer');
const researcherRoutes = require('./routes/researcher');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/farmer', farmerRoutes);
app.use('/api/researcher', researcherRoutes);

// Health check
app.get('/health', (_req, res) => {
	res.json({ status: 'ok' });
});

// Force reliable public DNS to resolve SRV records (fixes EREFUSED on some Windows networks)
try {
	dns.setServers(['8.8.8.8', '1.1.1.1']);
} catch (e) {
	console.warn('Could not set custom DNS servers:', e);
}

// MongoDB Connection
const mongoUri = process.env.MONGO_URI;
if (!mongoUri) {
	console.error('MONGO_URI is not set in environment variables');
	process.exit(1);
}

mongoose
	.connect(mongoUri)
	.then(() => {
		console.log('✅ Connected to MongoDB Atlas');
	})
	.catch((err) => {
		console.error('MongoDB connection error:', err);
		process.exit(1);
	});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
	console.log(`Server running on port ${PORT}`);
});


