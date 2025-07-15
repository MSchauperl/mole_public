#!/usr/bin/env python3
"""
Google Slides Presentation Generator for MolE Architecture
This script creates a complete Google Slides presentation for the MolE model.
"""

import json
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/presentations']

class MolEPresentationGenerator:
    def __init__(self):
        self.service = None
        self.presentation_id = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Google Slides API"""
        creds = None
        # The file token.json stores the user's access and refresh tokens.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('slides', 'v1', credentials=creds)
    
    def create_presentation(self, title="MolE: Molecular Embeddings with Disentangled Self-Attention"):
        """Create a new presentation"""
        try:
            body = {
                'title': title
            }
            presentation = self.service.presentations().create(body=body).execute()
            self.presentation_id = presentation['presentationId']
            print(f'Created presentation: {self.presentation_id}')
            return self.presentation_id
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None
    
    def get_slide_layouts(self):
        """Get available slide layouts"""
        try:
            presentation = self.service.presentations().get(
                presentationId=self.presentation_id).execute()
            return presentation.get('layouts', [])
        except HttpError as error:
            print(f'An error occurred: {error}')
            return []
    
    def add_slide(self, layout_id='BLANK'):
        """Add a new slide to the presentation"""
        try:
            requests = [{
                'createSlide': {
                    'objectId': f'slide_{len(self.get_slides()) + 1}',
                    'slideLayoutReference': {
                        'predefinedLayout': layout_id
                    }
                }
            }]
            
            body = {'requests': requests}
            response = self.service.presentations().batchUpdate(
                presentationId=self.presentation_id, body=body).execute()
            return response
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None
    
    def get_slides(self):
        """Get all slides in the presentation"""
        try:
            presentation = self.service.presentations().get(
                presentationId=self.presentation_id).execute()
            return presentation.get('slides', [])
        except HttpError as error:
            print(f'An error occurred: {error}')
            return []
    
    def add_text_to_slide(self, slide_id, text, position=None):
        """Add text to a slide"""
        try:
            # Default position if not specified
            if position is None:
                position = {
                    'x': {'magnitude': 50, 'unit': 'PT'},
                    'y': {'magnitude': 50, 'unit': 'PT'}
                }
            
            text_box_id = f'text_box_{slide_id}'
            requests = [
                {
                    'createShape': {
                        'objectId': text_box_id,
                        'shapeType': 'TEXT_BOX',
                        'elementProperties': {
                            'pageObjectId': slide_id,
                            'size': {
                                'height': {'magnitude': 400, 'unit': 'PT'},
                                'width': {'magnitude': 600, 'unit': 'PT'}
                            },
                            'transform': {
                                'scaleX': 1,
                                'scaleY': 1,
                                'translateX': position['x']['magnitude'],
                                'translateY': position['y']['magnitude'],
                                'unit': 'PT'
                            }
                        }
                    }
                },
                {
                    'insertText': {
                        'objectId': text_box_id,
                        'text': text
                    }
                }
            ]
            
            body = {'requests': requests}
            response = self.service.presentations().batchUpdate(
                presentationId=self.presentation_id, body=body).execute()
            return response
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None
    
    def create_mole_presentation(self):
        """Create the complete MolE presentation"""
        
        # Slide content data
        slides_data = [
            {
                "title": "MolE: Molecular Embeddings with Disentangled Self-Attention",
                "content": "A Transformer Architecture for Molecular Property Prediction",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "What is MolE?",
                "content": """• DeBERTa-inspired architecture with molecular adaptations
• Disentangled self-attention for enhanced molecular understanding
• Graph-aware processing that respects molecular structure
• Multi-task prediction capabilities

Key Innovation: Treats molecular graphs as sequences of atom environments while preserving structural information through advanced attention mechanisms.""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Core Architectural Innovations",
                "content": """1. Molecular Graph Integration
   • Converts molecular graphs to atom environment sequences
   • Preserves structural information through relative position embeddings
   • Handles variable-length molecules efficiently

2. Disentangled Self-Attention
   • Separates content (what atoms are) from position (where they are)
   • Four attention types: content-to-content, content-to-position, position-to-content, position-to-position

3. Pre-training Integration
   • Compatible with DeBERTa pre-trained weights
   • Transfer learning from language models to molecular domain""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Input Processing - From Molecules to Sequences",
                "content": """Challenge: How do you feed a molecular graph into a transformer?

MolE's Solution:
1. Molecular Graph Representation
   • Nodes: Atoms with features (element, charge, etc.)
   • Edges: Chemical bonds with attributes (bond type, distance)

2. Atom Environment Tokenization
   • Each atom becomes a token representing its local chemical environment
   • Environment includes atom type, bonding patterns, local structure

3. Sequence Formation
   • Molecular graph → Sequence of atom environment tokens
   • Variable length sequences (different molecule sizes)
   • Batch processing with padding""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Disentangled Self-Attention - Overview",
                "content": """What makes MolE's attention special?

Standard Transformer Attention:
• Only models content-to-content relationships
• Limited understanding of structural relationships

MolE's Disentangled Attention:
• Content: Chemical features of atoms (what they are)
• Position: Molecular structure/connectivity (where they are)
• Separation: Models content and position interactions independently

Four Attention Types:
1. c2c: Content-to-Content (how atoms attend to other atoms)
2. c2p: Content-to-Position (how atoms attend to molecular positions)
3. p2c: Position-to-Content (how positions attend to chemical features)
4. p2p: Position-to-Position (how molecular structure patterns interact)""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Embedding Layer Architecture",
                "content": """AtomEnvEmbeddings transforms molecular tokens into dense representations:

Components:
1. Word Embeddings
   • Maps atom environment tokens to dense vectors
   • Dimension: vocab_size × embedding_size

2. Position Embeddings
   • Encodes sequential position information
   • Dimension: max_position_embeddings × embedding_size

3. Type Embeddings
   • Handles different token types (primarily atoms)
   • Dimension: type_vocab_size × embedding_size

Process: Token → [Word Emb] + [Position Emb] + [Type Emb] → LayerNorm → Dropout → Output""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Encoder Architecture - Multi-Layer Transformer",
                "content": """BertEncoder Structure:
• Multi-layer transformer stack
• Configurable number of layers (typically 6-12)
• Gradient checkpointing support

BertLayer Components:
• Self-attention with disentangled mechanism
• Feed-forward network
• Residual connections
• Layer normalization

Key Features:
• Relative position embeddings for molecular graph structure
• Attention dropout for regularization
• Memory-efficient training options""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Prediction Heads - Multi-Task Architecture",
                "content": """TaskPredictionHead handles diverse molecular property prediction:

Architecture:
• Context token processing (CLS token equivalent)
• Dense layer → Activation → Dropout → Task-specific output

Capabilities:
• Multi-task support: Multiple properties simultaneously
• Classification & Regression: Flexible output types
• NaN handling: Robust training with missing labels

Applications:
• Molecular property prediction (LogP, solubility, toxicity)
• Drug discovery (ADMET prediction)
• Materials science (catalyst properties)""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Key Architectural Strengths",
                "content": """MolE's Unique Advantages:

1. Molecular Graph Awareness
   • Respects chemical structure through graph-aware attention
   • Preserves molecular connectivity information
   • Handles variable-length molecules efficiently

2. Advanced Attention Mechanisms
   • Disentangled attention for content/position separation
   • Multi-scale molecular understanding
   • Interpretable attention patterns

3. Pre-training Integration
   • Leverages powerful language model pre-training
   • Transfer learning from textual to molecular domains
   • Flexible architecture adaptation

4. Production-Ready Design
   • Robust error handling and logging
   • Distributed training support
   • Comprehensive configuration system""",
                "layout": "TITLE_AND_BODY"
            },
            {
                "title": "Summary & Conclusion",
                "content": """MolE: Key Takeaways

Innovation:
• First transformer architecture with molecular graph-aware disentangled attention
• Separates chemical content from structural position information
• Leverages pre-trained language models for molecular understanding

Technical Excellence:
• Robust, production-ready architecture
• Flexible configuration system
• Comprehensive training infrastructure

Impact:
• State-of-the-art molecular property prediction
• Enables multi-task learning across diverse chemical properties
• Provides interpretable insights into molecular behavior

Future: MolE represents a significant advance in molecular machine learning, combining the power of transformers with domain-specific molecular understanding.""",
                "layout": "TITLE_AND_BODY"
            }
        ]
        
        # Create presentation
        presentation_id = self.create_presentation()
        if not presentation_id:
            return None
        
        # Get the first slide (automatically created)
        slides = self.get_slides()
        if slides:
            # Update the first slide with title slide content
            first_slide = slides[0]
            self.add_text_to_slide(
                first_slide['objectId'], 
                slides_data[0]['title'],
                {'x': {'magnitude': 100, 'unit': 'PT'}, 'y': {'magnitude': 100, 'unit': 'PT'}}
            )
            self.add_text_to_slide(
                first_slide['objectId'], 
                slides_data[0]['content'],
                {'x': {'magnitude': 100, 'unit': 'PT'}, 'y': {'magnitude': 200, 'unit': 'PT'}}
            )
        
        # Add remaining slides
        for i, slide_data in enumerate(slides_data[1:], 1):
            # Add new slide
            self.add_slide('TITLE_AND_BODY')
            
            # Get updated slides list
            slides = self.get_slides()
            if len(slides) > i:
                current_slide = slides[i]
                
                # Add title
                self.add_text_to_slide(
                    current_slide['objectId'], 
                    slide_data['title'],
                    {'x': {'magnitude': 50, 'unit': 'PT'}, 'y': {'magnitude': 50, 'unit': 'PT'}}
                )
                
                # Add content
                self.add_text_to_slide(
                    current_slide['objectId'], 
                    slide_data['content'],
                    {'x': {'magnitude': 50, 'unit': 'PT'}, 'y': {'magnitude': 150, 'unit': 'PT'}}
                )
        
        print(f"✅ Created MolE presentation with {len(slides_data)} slides")
        print(f"📊 Presentation ID: {presentation_id}")
        print(f"🔗 View at: https://docs.google.com/presentation/d/{presentation_id}")
        
        return presentation_id

def main():
    """Main function to create the MolE presentation"""
    print("🚀 Creating MolE Architecture Presentation...")
    
    # Check for credentials file
    if not os.path.exists('credentials.json'):
        print("❌ Error: credentials.json not found!")
        print("📝 Please follow these steps:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing")
        print("3. Enable Google Slides API")
        print("4. Create OAuth2 credentials")
        print("5. Download as 'credentials.json'")
        return
    
    try:
        generator = MolEPresentationGenerator()
        presentation_id = generator.create_mole_presentation()
        
        if presentation_id:
            print(f"\n🎉 Success! Your MolE presentation is ready!")
            print(f"📱 You can now:")
            print(f"   • View: https://docs.google.com/presentation/d/{presentation_id}")
            print(f"   • Edit: Add diagrams from the generated images")
            print(f"   • Share: Send the link to collaborators")
        else:
            print("❌ Failed to create presentation")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔍 Make sure you have proper Google API credentials set up")

if __name__ == '__main__':
    main()