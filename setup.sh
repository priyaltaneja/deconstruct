#!/bin/bash
# Deconstruct Setup Script
# Automates initial setup for all components

set -e

echo "🚀 Deconstruct Setup"
echo "===================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check Node
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found${NC}"

# Check Supabase CLI
if ! command -v supabase &> /dev/null; then
    echo -e "${YELLOW}⚠ Supabase CLI not found. Install with: brew install supabase/tap/supabase${NC}"
    read -p "Continue without Supabase? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Supabase CLI found${NC}"
fi

# Check Modal
if ! command -v modal &> /dev/null; then
    echo -e "${YELLOW}⚠ Modal CLI not found. Will install via pip.${NC}"
fi

echo ""
echo "🔧 Setting up components..."
echo ""

# 1. GPU Setup
echo "1️⃣ Setting up GPU (Modal)..."
cd gpu
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if ! command -v modal &> /dev/null; then
    echo -e "${YELLOW}⚠ Modal not found in PATH after install. Trying to authenticate...${NC}"
fi

echo "   To authenticate with Modal, run: modal token set"
echo "   To create API key secret, run: modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-..."
deactivate
cd ..
echo -e "${GREEN}✓ GPU setup complete${NC}"
echo ""

# 2. Web Setup
echo "2️⃣ Setting up Web (React)..."
cd web
npm install --silent
echo -e "${GREEN}✓ Web setup complete${NC}"
cd ..
echo ""

# 3. Supabase Setup
if command -v supabase &> /dev/null; then
    echo "3️⃣ Setting up Supabase..."
    cd supabase

    # Check if Supabase is already running
    if supabase status &> /dev/null; then
        echo "   Supabase is already running"
    else
        echo "   Starting Supabase (this may take a minute)..."
        supabase start
    fi

    echo ""
    echo "   Supabase credentials:"
    supabase status | grep -E "(API URL|anon key)"
    echo ""

    cd ..
    echo -e "${GREEN}✓ Supabase setup complete${NC}"
else
    echo -e "${YELLOW}⚠ Skipping Supabase setup${NC}"
fi
echo ""

# 4. Create .env files
echo "4️⃣ Creating environment files..."

# GPU .env
if [ ! -f "gpu/.env" ]; then
    cat > gpu/.env << EOF
# Modal Configuration
MODAL_WORKSPACE=your-workspace

# API Keys (also add to Modal secrets)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Supabase
SUPABASE_URL=http://localhost:54321
SUPABASE_ANON_KEY=your-anon-key
EOF
    echo -e "${GREEN}✓ Created gpu/.env${NC}"
else
    echo "   gpu/.env already exists"
fi

# Web .env
if [ ! -f "web/.env" ]; then
    cat > web/.env << EOF
# Supabase
VITE_SUPABASE_URL=http://localhost:54321
VITE_SUPABASE_ANON_KEY=your-anon-key

# API Endpoint (Modal)
VITE_API_URL=http://localhost:8000
EOF
    echo -e "${GREEN}✓ Created web/.env${NC}"
else
    echo "   web/.env already exists"
fi
echo ""

echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo ""
echo "1. Add your Anthropic API key to Modal:"
echo "   cd gpu && source venv/bin/activate"
echo "   modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "2. Deploy to Modal:"
echo "   modal deploy modal_app.py"
echo ""
echo "3. Start the web dashboard:"
echo "   cd web && npm run dev"
echo ""
echo "4. Open http://localhost:3000 and start uploading documents!"
echo ""
echo "For detailed instructions, see README.md and CLAUDE.md"
echo ""
