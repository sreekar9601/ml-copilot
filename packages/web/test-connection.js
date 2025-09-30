// Simple test script to verify API connection
const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  console.error('❌ NEXT_PUBLIC_API_URL environment variable is required');
  process.exit(1);
}

async function testConnection() {
  console.log('🔍 Testing API connection...');
  console.log('📍 API URL:', API_URL);
  
  try {
    // Test health endpoint
    const healthResponse = await fetch(`${API_URL}/health`);
    const healthData = await healthResponse.json();
    console.log('✅ Health check:', healthData);
    
    // Test stats endpoint
    const statsResponse = await fetch(`${API_URL}/stats`);
    const statsData = await statsResponse.json();
    console.log('📊 Stats:', statsData);
    
    // Test a simple query
    const queryResponse = await fetch(`${API_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        q: 'What is PyTorch DataLoader?',
        top_k: 3,
        include_sources: true
      })
    });
    
    if (queryResponse.ok) {
      const queryData = await queryResponse.json();
      console.log('✅ Query test successful');
      console.log('📝 Answer preview:', queryData.answer.substring(0, 100) + '...');
      console.log('📚 Sources found:', queryData.sources.length);
    } else {
      console.log('❌ Query test failed:', queryResponse.status, queryResponse.statusText);
    }
    
  } catch (error) {
    console.error('❌ Connection failed:', error.message);
  }
}

testConnection();
