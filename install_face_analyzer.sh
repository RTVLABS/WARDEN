#!/bin/bash

# Text colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Installing Face Analyzer...${NC}"

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
python3 -m venv .venv

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}📥 Installing dependencies...${NC}"
pip install opencv-python
pip install ultralytics
pip install deepface
pip install numpy

# Create run script
echo -e "${BLUE}📝 Creating run script...${NC}"
cat > run.sh << 'EOL'
#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run face analyzer with provided arguments
python face_analyzer.py "$@"
EOL

# Make run script executable
chmod +x run.sh

echo -e "${GREEN}✅ Installation complete!${NC}"
echo -e "${BLUE}To run Face Analyzer:${NC}"
echo -e "1. Basic usage: ${GREEN}./run.sh${NC}"
echo -e "2. With options: ${GREEN}./run.sh -v -c 0.8${NC}"
echo -e "3. For help: ${GREEN}./run.sh --help${NC}" 