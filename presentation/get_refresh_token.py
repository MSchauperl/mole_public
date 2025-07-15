#!/usr/bin/env python3
"""
Script to get Google OAuth refresh token for MCP server
Run this once to get the refresh token you need for the MCP server.
"""

import json
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/presentations']

def get_refresh_token():
    """Get refresh token for Google Slides API"""
    
    # Check if cursor_googleslides_api.json exists
    if not os.path.exists('/home/mschauperl/keys/cursor_googleslides_api.json'):
        print("❌ Error: cursor_googleslides_api.json not found!")
        print("📝 Please download OAuth 2.0 credentials from Google Cloud Console")
        print("   1. Go to console.cloud.google.com")
        print("   2. APIs & Services → Credentials")
        print("   3. Create Credentials → OAuth client ID → Desktop application")
        print("   4. Download as 'cursor_googleslides_api.json'")
        return None
    
    print("🔐 Starting OAuth flow to get refresh token...")
    
    # Run OAuth flow
    flow = InstalledAppFlow.from_client_secrets_file('/home/mschauperl/keys/cursor_googleslides_api.json', SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Extract credentials
    with open('/home/mschauperl/keys/cursor_googleslides_api.json', 'r') as f:
        client_info = json.load(f)
    
    client_id = client_info['installed']['client_id']
    client_secret = client_info['installed']['client_secret']
    refresh_token = creds.refresh_token
    
    print("\n✅ Success! Here are your MCP server environment variables:")
    print("=" * 60)
    print(f"export GOOGLE_CLIENT_ID='{client_id}'")
    print(f"export GOOGLE_CLIENT_SECRET='{client_secret}'")
    print(f"export GOOGLE_REFRESH_TOKEN='{refresh_token}'")
    print("=" * 60)
    
    # Save to .env file for convenience
    with open('.env', 'w') as f:
        f.write(f"GOOGLE_CLIENT_ID={client_id}\n")
        f.write(f"GOOGLE_CLIENT_SECRET={client_secret}\n")
        f.write(f"GOOGLE_REFRESH_TOKEN={refresh_token}\n")
    
    print("\n💾 Also saved to .env file")
    print("🚀 You can now run the MCP server!")
    
    return {
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token
    }

if __name__ == '__main__':
    get_refresh_token()
