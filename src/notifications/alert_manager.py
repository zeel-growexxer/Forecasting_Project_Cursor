import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """Manages notifications and alerts for the forecasting pipeline"""
    
    def __init__(self, config_path='config.ini'):
        self.config = self._load_config(config_path)
        self.notification_history = []
    
    def _load_config(self, config_path):
        """Load notification configuration"""
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Default notification settings
            notification_config = {
                'email_enabled': config.getboolean('notifications', 'email_enabled', fallback=False),
                'slack_enabled': config.getboolean('notifications', 'slack_enabled', fallback=False),
                'smtp_server': config.get('notifications', 'smtp_server', fallback='smtp.gmail.com'),
                'smtp_port': config.getint('notifications', 'smtp_port', fallback=587),
                'email_from': config.get('notifications', 'email_from', fallback=''),
                'email_password': config.get('notifications', 'email_password', fallback=''),
                'email_to': config.get('notifications', 'email_to', fallback=''),
                'slack_webhook': config.get('notifications', 'slack_webhook', fallback=''),
                'performance_threshold': config.getfloat('notifications', 'performance_threshold', fallback=0.8)
            }
            return notification_config
        except Exception as e:
            logger.warning(f"Could not load notification config: {e}")
            return {}
    
    def send_email_alert(self, subject: str, message: str, priority: str = 'normal'):
        """Send email alert"""
        if not self.config.get('email_enabled', False):
            logger.info("Email notifications disabled")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email_from']
            msg['To'] = self.config['email_to']
            msg['Subject'] = f"[{priority.upper()}] {subject}"
            
            body = f"""
            <html>
            <body>
                <h2>{subject}</h2>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Priority:</strong> {priority.upper()}</p>
                <hr>
                <pre>{message}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email_from'], self.config['email_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_slack_alert(self, message: str, channel: str = '#alerts', priority: str = 'normal'):
        """Send Slack alert"""
        if not self.config.get('slack_enabled', False):
            logger.info("Slack notifications disabled")
            return False
        
        try:
            color = {
                'low': '#36a64f',      # Green
                'normal': '#ff9500',   # Orange
                'high': '#ff0000'      # Red
            }.get(priority, '#ff9500')
            
            payload = {
                "channel": channel,
                "attachments": [{
                    "color": color,
                    "title": f"Forecasting Pipeline Alert - {priority.upper()}",
                    "text": message,
                    "footer": f"Sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
                }]
            }
            
            response = requests.post(self.config['slack_webhook'], json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def pipeline_success_alert(self, duration: float, models_trained: List[str]):
        """Alert for successful pipeline completion"""
        message = f"""
        🎉 Pipeline completed successfully!
        
        Duration: {duration:.2f} minutes
        Models trained: {', '.join(models_trained)}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_email_alert("Pipeline Success", message, 'low')
        self.send_slack_alert(message, priority='low')
    
    def pipeline_failure_alert(self, error: str, step: str):
        """Alert for pipeline failure"""
        message = f"""
        ❌ Pipeline failed!
        
        Failed step: {step}
        Error: {error}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_email_alert("Pipeline Failure", message, 'high')
        self.send_slack_alert(message, priority='high')
    
    def performance_alert(self, model_name: str, metric: str, value: float, threshold: float):
        """Alert for poor model performance"""
        message = f"""
        ⚠️ Model Performance Alert
        
        Model: {model_name}
        Metric: {metric}
        Current value: {value:.4f}
        Threshold: {threshold:.4f}
        Status: {'Below threshold' if value < threshold else 'Above threshold'}
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        priority = 'high' if value < threshold else 'normal'
        self.send_email_alert("Performance Alert", message, priority)
        self.send_slack_alert(message, priority=priority)
    
    def data_quality_alert(self, issues: List[str]):
        """Alert for data quality issues"""
        message = f"""
        🔍 Data Quality Issues Detected
        
        Issues found:
        {chr(10).join(f'• {issue}' for issue in issues)}
        
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_email_alert("Data Quality Alert", message, 'normal')
        self.send_slack_alert(message, priority='normal')
    
    def log_notification(self, notification_type: str, message: str, success: bool):
        """Log notification for audit trail"""
        notification = {
            'timestamp': datetime.now().isoformat(),
            'type': notification_type,
            'message': message[:100] + '...' if len(message) > 100 else message,
            'success': success
        }
        self.notification_history.append(notification)
        
        # Keep only last 100 notifications
        if len(self.notification_history) > 100:
            self.notification_history = self.notification_history[-100:]

# Global alert manager instance
alert_manager = AlertManager() 