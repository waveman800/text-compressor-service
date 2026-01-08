#!/usr/bin/env python3
"""Test script to verify the server is running"""

import requests
import json

def test_server():
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Server is running!")
            print("Health check response:", response.json())
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
        
        # Test models endpoint
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            print("\n📊 Available models:")
            models = response.json()
            for model in models['models']:
                print(f"  - {model['id']}: {model['name']} (context: {model['context_length']})")
        else:
            print(f"❌ Models endpoint returned status code: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_server()