#!/usr/bin/env python3
"""
Simple script to open the dashboard in the browser
"""
import webbrowser
import time
import subprocess
import sys
import os

def main():
    """Open the dashboard"""
    print("🚀 Opening Forecasting Pipeline Dashboard...")
    
    # Check if dashboard is running
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=2)
        if response.status_code == 200:
            print("✅ Dashboard is running at http://localhost:8501")
            webbrowser.open("http://localhost:8501")
            return
    except:
        pass
    
    # If not running, start it
    print("🔄 Starting dashboard...")
    
    # Get the dashboard path
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "src", 
        "dashboard", 
        "dashboard_simple.py"
    )
    
    try:
        # Start the dashboard in background
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        # Wait a moment for it to start
        print("⏳ Waiting for dashboard to start...")
        time.sleep(5)
        
        # Open in browser
        print("🌐 Opening dashboard in browser...")
        webbrowser.open("http://localhost:8501")
        
        print("✅ Dashboard should now be open in your browser!")
        print("📍 URL: http://localhost:8501")
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Try running manually: streamlit run src/dashboard/dashboard_simple.py")

if __name__ == "__main__":
    main() 