const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const router = express.Router();

// POST /api/auth/signup
router.post('/signup', async (req, res) => {
	try {
		const { username, password } = req.body;

		if (!username || !password) {
			return res.status(400).json({ success: false, message: 'Username and password are required' });
		}

		const existing = await User.findOne({ username });
		if (existing) {
			return res.status(409).json({ success: false, message: 'Username already exists' });
		}

		const salt = await bcrypt.genSalt(10);
		const hashedPassword = await bcrypt.hash(password, salt);

		const user = new User({ username, password: hashedPassword });
		await user.save();

		return res.status(201).json({ success: true, message: 'User created successfully' });
	} catch (error) {
		console.error('Signup error:', error);
		return res.status(500).json({ success: false, message: 'Internal server error' });
	}
});

// POST /api/auth/signin
router.post('/signin', async (req, res) => {
	try {
		const { username, password } = req.body;

		if (!username || !password) {
			return res.status(400).json({ success: false, message: 'Username and password are required' });
		}

		const user = await User.findOne({ username });
		if (!user) {
			return res.status(401).json({ success: false, message: 'Invalid credentials' });
		}

		const isMatch = await bcrypt.compare(password, user.password);
		if (!isMatch) {
			return res.status(401).json({ success: false, message: 'Invalid credentials' });
		}

		const payload = { id: user._id.toString(), username: user.username };
		const token = jwt.sign(payload, process.env.JWT_SECRET || 'randomSecretKey', { expiresIn: '7d' });

		return res.json({ success: true, token, username: user.username });
	} catch (error) {
		console.error('Signin error:', error);
		return res.status(500).json({ success: false, message: 'Internal server error' });
	}
});

module.exports = router;


