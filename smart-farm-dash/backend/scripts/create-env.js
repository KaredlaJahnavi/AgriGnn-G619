const fs = require('fs');
const path = require('path');

const envPath = path.join(__dirname, '..', '.env');

const content = [
  'PORT=5000',
  'MONGO_URI=mongodb+srv://agrignn_user:agri_gnn@agrignncluster.cz04xxj.mongodb.net/?retryWrites=true&w=majority&appName=AgriGNNCluster',
  'JWT_SECRET=randomSecretKey',
  ''
].join('\n');

if (fs.existsSync(envPath)) {
  console.log('.env already exists, skipping creation.');
  process.exit(0);
}

fs.writeFileSync(envPath, content, { encoding: 'utf8' });
console.log('Created backend/.env');


