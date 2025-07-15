# Google Slides Creation Guide for MolE Presentation

## Overview
This guide provides multiple options for creating your MolE presentation in Google Slides, including using MCP servers or a custom Python script.

## Option 1: Using MCP Server (Recommended for AI Integration)

### Available MCP Servers
1. **matteoantoci/google-slides-mcp** - Full-featured GitHub-based server
2. **Pipedream Google Slides MCP** - Cloud-based with automation
3. **Zapier MCP AI** - Integration with other tools

### Setup: matteoantoci/google-slides-mcp

#### Step 1: Install the MCP Server
```bash
# Clone the repository
git clone https://github.com/matteoantoci/google-slides-mcp.git
cd google-slides-mcp

# Install dependencies
npm install
```

#### Step 2: Set up Google API Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Slides API:
   - Navigate to "APIs & Services" → "Library"
   - Search for "Google Slides API"
   - Click "Enable"
4. Create OAuth2 credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Choose "Desktop application"
   - Download the credentials JSON file

#### Step 3: Configure the Server
```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
export CLIENT_ID="your-client-id"
export CLIENT_SECRET="your-client-secret"
```

#### Step 4: Build and Run
```bash
# Build the server
npm run build

# Start the server
npm start
```

#### Step 5: Connect to Your AI Assistant
The MCP server will expose tools like:
- `create_presentation`
- `get_presentation`
- `add_slide`
- `update_slide`
- `insert_text`
- `insert_image`

## Option 2: Using Python Script (Ready to Use)

### Quick Setup

#### Step 1: Install Dependencies
```bash
# Navigate to the presentation folder
cd mole_public/presentation

# Install required packages
pip install -r requirements.txt
```

#### Step 2: Set up Google API Credentials
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Slides API
4. Create OAuth2 credentials for "Desktop application"
5. Download credentials as `credentials.json` in the presentation folder

#### Step 3: Run the Script
```bash
python create_google_slides.py
```

The script will:
- Create a new Google Slides presentation
- Add all 10 main slides with content
- Provide a link to view/edit the presentation

#### Step 4: Add Diagrams
1. Generate images from the Mermaid diagrams (see `diagrams/README.md`)
2. Upload images to the appropriate slides
3. Adjust formatting as needed

### Script Features
- **Automated slide creation**: Creates all slides with proper titles and content
- **Structured content**: Organizes information logically
- **Easy customization**: Modify slide content in the script
- **Error handling**: Robust error handling and user feedback

## Option 3: Manual Creation with Generated Content

### Quick Start
1. Open [Google Slides](https://slides.google.com/)
2. Create a new presentation
3. Use the content from `mole_presentation.md`
4. Add diagrams generated from the `.mmd` files

### Content Organization
- **Title Slide**: Use slide 1 content
- **Main Slides**: Copy content from slides 2-24
- **Technical Details**: Use as speaker notes or additional slides
- **Diagrams**: Generate from Mermaid files and insert

## Comparison of Options

| Feature | MCP Server | Python Script | Manual |
|---------|------------|---------------|---------|
| **Setup Complexity** | Medium | Low | None |
| **Automation Level** | High | Medium | Low |
| **Customization** | High | Medium | High |
| **AI Integration** | Excellent | None | None |
| **Maintenance** | Medium | Low | None |
| **Real-time Updates** | Yes | No | No |

## Recommended Workflow

### For AI-Assisted Creation (Best Experience)
1. Set up the MCP server
2. Use AI assistant to create and modify slides
3. Add diagrams through the assistant
4. Fine-tune formatting manually

### For Quick Creation
1. Use the Python script to generate basic structure
2. Generate diagrams from Mermaid files
3. Add diagrams to slides manually
4. Customize formatting and styling

### For Complete Control
1. Create presentation manually
2. Use provided content as reference
3. Generate and add diagrams
4. Customize extensively

## Tips for Success

### Content Integration
- **Use provided content**: All slide content is ready in `mole_presentation.md`
- **Add diagrams**: Generate high-quality images from Mermaid files
- **Include speaker notes**: Use technical details for comprehensive notes
- **Maintain consistency**: Use uniform formatting and styling

### Visual Enhancement
- **Generate diagrams**: Use the `.mmd` files to create professional diagrams
- **Consistent colors**: Use the color scheme from `visual_guide.md`
- **High-quality images**: Generate at least 1920x1080 resolution
- **Professional layout**: Follow the visual guidelines provided

### Technical Accuracy
- **Verify code examples**: All code is from the actual MolE implementation
- **Check mathematical formulas**: Formulations match the code
- **Update configurations**: Modify based on your specific needs

## Troubleshooting

### Common Issues

#### Authentication Problems
- **Issue**: "Authentication failed"
- **Solution**: Ensure credentials.json is in the correct location
- **Check**: Verify Google Slides API is enabled

#### Permission Errors
- **Issue**: "Insufficient permissions"
- **Solution**: Use OAuth2 credentials, not service account
- **Check**: Ensure correct scopes are requested

#### Script Errors
- **Issue**: "Module not found"
- **Solution**: Install requirements: `pip install -r requirements.txt`
- **Check**: Python version compatibility (3.7+)

### Support Resources
- **Google Slides API Documentation**: https://developers.google.com/slides
- **MCP Server GitHub**: https://github.com/matteoantoci/google-slides-mcp
- **Python Google API Client**: https://github.com/googleapis/google-api-python-client

## Next Steps

1. **Choose your approach**: MCP server, Python script, or manual
2. **Set up credentials**: Follow the Google API setup steps
3. **Generate diagrams**: Use the Mermaid files to create visuals
4. **Create presentation**: Run your chosen method
5. **Customize**: Add diagrams and fine-tune formatting
6. **Review**: Check all content for accuracy and completeness

Your MolE presentation will be ready for technical conferences, research presentations, or educational purposes!