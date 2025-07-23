#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import time
import logging
import signal
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedPipeline:
    def __init__(self):
        self.processes = []
        self.running = False
        
    def start_prefect_server(self):
        """Start Prefect server"""
        try:
            logger.info("Starting Prefect server...")
            process = subprocess.Popen([
                "prefect", "server", "start"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(("Prefect Server", process))
            time.sleep(5)  # Wait for server to start
            return True
        except Exception as e:
            logger.error(f"Failed to start Prefect server: {e}")
            return False
    
    def start_prefect_agent(self):
        """Start Prefect agent"""
        try:
            logger.info("Starting Prefect agent...")
            process = subprocess.Popen([
                "prefect", "agent", "start", "-p", "default"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(("Prefect Agent", process))
            return True
        except Exception as e:
            logger.error(f"Failed to start Prefect agent: {e}")
            return False
    
    def create_deployment(self):
        """Create Prefect deployment"""
        try:
            logger.info("Creating Prefect deployment...")
            
            # For now, let's skip the deployment creation and just run the flow directly
            # This will be handled by the data monitor instead
            logger.info("Skipping deployment creation - will use data monitor for scheduling")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            return False
    
    def start_data_monitor(self):
        """Start data monitoring service"""
        try:
            logger.info("Starting data monitor...")
            
            from src.data.monitor import DataMonitor
            
            def monitor_loop():
                monitor = DataMonitor()
                last_daily_run = None
                while self.running:
                    try:
                        current_time = time.time()
                        
                        # Check for new data every 5 minutes
                        if monitor.check_for_new_data():
                            if monitor.validate_raw_data():
                                logger.info("New data detected, triggering pipeline...")
                                monitor.trigger_preprocessing()
                                # Trigger retraining
                                from src.pipeline.retrain_flow import retrain_all_models
                                retrain_all_models()
                        
                        # Check for daily run at 2 AM
                        if last_daily_run is None or (current_time - last_daily_run) > 86400:  # 24 hours
                            from datetime import datetime
                            now = datetime.now()
                            if now.hour == 2 and now.minute < 5:  # Between 2:00-2:05 AM
                                logger.info("Daily 2 AM retraining triggered...")
                                from src.pipeline.retrain_flow import retrain_all_models
                                retrain_all_models()
                                last_daily_run = current_time
                        
                    except Exception as e:
                        logger.error(f"Data monitor error: {e}")
                    time.sleep(300)  # Check every 5 minutes
            
            monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            monitor_thread.start()
            self.processes.append(("Data Monitor", monitor_thread))
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data monitor: {e}")
            return False
    
    def start_services(self):
        """Start all automated pipeline services"""
        logger.info("🚀 Starting automated forecasting pipeline...")
        
        # Start Prefect server
        if not self.start_prefect_server():
            return False
        
        # Create deployment
        if not self.create_deployment():
            return False
        
        # Start Prefect agent
        if not self.start_prefect_agent():
            return False
        
        # Start data monitor
        if not self.start_data_monitor():
            return False
        
        self.running = True
        logger.info("✅ Automated pipeline started successfully!")
        logger.info("📊 Dashboard: http://localhost:8501")
        logger.info("🔧 MLflow UI: http://localhost:5000")
        logger.info("⏰ Retraining scheduled: Daily at 2 AM UTC")
        logger.info("👀 Data monitoring: Every 5 minutes")
        logger.info("Press Ctrl+C to stop the pipeline")
        
        return True
    
    def stop_services(self):
        """Stop all services"""
        logger.info("🛑 Stopping automated pipeline...")
        self.running = False
        
        for name, process in self.processes:
            try:
                if hasattr(process, 'terminate'):
                    process.terminate()
                    logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        self.processes.clear()
        logger.info("Pipeline stopped")

def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    logger.info("Received interrupt signal")
    if pipeline:
        pipeline.stop_services()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start pipeline
    pipeline = AutomatedPipeline()
    
    if pipeline.start_services():
        try:
            # Keep running
            while pipeline.running:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.stop_services()
    else:
        logger.error("Failed to start automated pipeline")
        sys.exit(1) 