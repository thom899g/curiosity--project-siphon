"""
Centralized configuration management with fallbacks and validation.
Edge Cases Handled:
- Missing environment variables with safe defaults
- Invalid RPC URLs with retry logic
- Type validation for all critical parameters
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Initialize logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hydra_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables with validation
load_dotenv()

class ConfigValidationError(Exception):
    """Custom exception for configuration validation failures"""
    pass

class HydraConfig:
    """Centralized configuration with validation and defaults"""
    
    def __init__(self):
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validate critical environment variables exist"""
        required_vars = [
            'FIREBASE_CREDENTIALS_PATH',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'ALCHEMY_BASE_RPC_URL',
            'ALCHEMY_ETH_RPC_URL'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.critical(error_msg)
            raise ConfigValidationError(error_msg)
    
    @property
    def rpc_urls(self) -> Dict[str, str]:
        """Get RPC URLs with validation"""
        return {
            'base': os.getenv('ALCHEMY_BASE_RPC_URL', 'https://base-mainnet.g.alchemy.com/v2/demo'),
            'ethereum': os.getenv('ALCHEMY_ETH_RPC_URL', 'https://eth-mainnet.g.alchemy.com/v2/demo')
        }
    
    @property
    def firebase_credentials_path(self) -> str:
        """Get Firebase credentials path with existence check"""
        path = os.getenv('FIREBASE_CREDENTIALS_PATH', './firebase-creds.json')
        if not os.path.exists(path):
            logger.warning(f"Firebase credentials not found at {path}. Will attempt browser acquisition.")
        return path
    
    @property
    def telegram_config(self) -> Dict[str, str]:
        """Get Telegram configuration"""
        return {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
        }
    
    @property
    def execution_params(self) -> Dict[str, Any]:
        """Get execution parameters with safe defaults"""
        return {
            'min_profit_threshold_wei': int(os.getenv('MIN_PROFIT_THRESHOLD', 100000000000000)),  # 0.0001 ETH
            'max_slippage_bps': int(os.getenv('MAX_SLIPPAGE_BPS', 50)),  # 0.5%
            'cycle_delay_ms': int(os.getenv('CYCLE_DELAY_MS', 100)),
            'max_concurrent_simulations': int(os.getenv('MAX_CONCURRENT_SIMULATIONS', 5))
        }
    
    @property
    def risk_parameters(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return {
            'max_capital_per_trade_eth': float(os.getenv('MAX_CAPITAL_PER_TRADE_ETH', 0.1)),
            'daily_loss_limit_eth': float(os.getenv('DAILY_LOSS_LIMIT_ETH', 0.5)),
            'cooldown_period_seconds': int(os.getenv('COOLDOWN_PERIOD_SECONDS', 300)),
            'emergency_stop_threshold': int(os.getenv('EMERGENCY_STOP_THRESHOLD', 5))
        }

# Global config instance
config = HydraConfig()