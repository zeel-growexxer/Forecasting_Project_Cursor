#!/usr/bin/env python3
"""
Script to run the forecasting pipeline dashboard
"""
import sys
import os
import subprocess
import webbrowser
import time

def main():
    """Run the dashboard"""
    print("🚀 Starting Forecasting Pipeline Dashboard...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        import plotly
    except ImportError:
        print("❌ Missing required packages. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Get the dashboard path
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", 
        "src", 
        "dashboard", 
        "dashboard.py"
    )
    
    # Start the dashboard
    print(f"📊 Dashboard will be available at: http://localhost:8501")
    print("🔄 Starting Streamlit server...")
    
    try:
        # Check if port 8501 is already in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8501))
        sock.close()
        
        if result == 0:
            print("⚠️  Port 8501 is already in use. Dashboard may already be running.")
            print("🌐 Opening existing dashboard in browser...")
            webbrowser.open("http://localhost:8501")
            return
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main() 